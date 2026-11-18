"""
subtree_partition
=================
Split a rooted binary phylogeny (as stored by phyddle's Format step) into
up to `n_parts` subtrees, built left to right. The network then mean-pools
node embeddings within each subtree.

Node ids are 0-based row positions in x (ape node number - 1).
edge_index is the symmetric [2, 2E] tensor/array built in HDF5BlockDataset,
whose first E columns are one direction of each edge.
"""
import numpy as np


def build_tree(edge_index, n_nodes):
    """Return (parent, children, root) from a symmetric edge_index.

    The first half of edge_index is tried as parent->child, then as
    child->parent; whichever gives exactly one root is used. children[v]
    keeps edge order, so children[v][0] is the left child.
    """
    ei = np.asarray(edge_index)
    E = ei.shape[1] // 2
    for src, dst in ((ei[0, :E], ei[1, :E]), (ei[1, :E], ei[0, :E])):
        parent = np.full(n_nodes, -1, dtype=np.int64)
        children = [[] for _ in range(n_nodes)]
        for p, c in zip(src.tolist(), dst.tolist()):
            parent[c] = p
            children[p].append(c)
        roots = np.flatnonzero(parent == -1)
        if len(roots) == 1:
            return parent, children, int(roots[0])
    raise ValueError("Could not find a unique root in edge_index")


def preorder(children, root):
    """Nodes in preorder, left child first."""
    order, stack = [], [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(reversed(children[v]))  # left child popped first
    return order


def tips_left_to_right(children, root):
    """Tips in the order reached by always going left first from the root."""
    return [v for v in preorder(children, root) if not children[v]]


def _unvisited_below(v, children, visited):
    """v plus all of its unvisited descendants (v itself assumed unvisited)."""
    out, stack = [v], [c for c in children[v] if not visited[c]]
    while stack:
        u = stack.pop()
        out.append(u)
        stack.extend(c for c in children[u] if not visited[c])
    return out


def partition_subtrees(parent, children, root, n_parts=10, max_overshoot=0.6):
    """Greedy left-to-right subtree partition with a shrinking size rule.

    Subtree k (0-based) aims to reach at least
        min_size = (nodes not yet visited) / (n_parts - k)
    nodes. Starting from the leftmost unvisited tip, it climbs one level at
    a time; each climb adds the parent and all of the parent's unvisited
    descendants.

    A climb is rejected, and the subtree stops where it is, if it would make
    the subtree larger than (1 + max_overshoot) * min_size. The nodes it
    would have taken stay unvisited for later subtrees. A subtree must hold
    more than one node, so the first climb from a tip is never rejected.
    The last subtree (k = n_parts - 1) never rejects a climb, so it goes to
    the root and takes everything still unvisited.

    If the tips run out before n_parts subtrees are built, the remaining
    unvisited nodes (ancestors of finished subtrees) are assigned bottom-up,
    each joining the smaller of its children's subtrees, so every subtree
    stays connected.

    Finally, any subtree that still has only one node (possible only for the
    last subtree in edge cases) is merged into the smallest neighbouring
    subtree, and ids are renumbered 0..k-1.

    Returns
        part : int64 array, subtree id (0..k-1) for each node
        k    : number of subtrees built
    """
    n = len(parent)
    part = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    sizes = []
    k = 0
    for tip in tips_left_to_right(children, root):
        if visited[tip]:
            continue
        if k == n_parts:
            break
        min_size = (~visited).sum() / (n_parts - k)
        max_size = (1 + max_overshoot) * min_size
        last = (k == n_parts - 1)
        members = [tip]
        visited[tip] = True
        cur = tip
        while len(members) < min_size and cur != root:
            nxt = int(parent[cur])
            # A visited node always has all its descendants visited, so an
            # ancestor of an unvisited tip is itself unvisited.
            assert not visited[nxt]
            gain = _unvisited_below(nxt, children, visited)
            if (not last and len(members) > 1
                    and len(members) + len(gain) > max_size):
                break                      # climb would overshoot: stop here
            visited[gain] = True
            members.extend(gain)
            cur = nxt
        part[members] = k
        sizes.append(len(members))
        k += 1

    # Tips ran out before n_parts: give each leftover ancestor (children
    # first) to the smaller of its children's subtrees.
    for v in reversed(preorder(children, root)):
        if part[v] == -1:
            best = min((part[c] for c in children[v]), key=lambda p: sizes[p])
            part[v] = best
            sizes[best] += 1
    return _merge_singletons(part, parent, children)


def _merge_singletons(part, parent, children):
    """Merge every one-node subtree into its smallest neighbouring subtree
    (parent's or a child's), then renumber ids 0..k-1 keeping their order."""
    if len(part) < 2:
        return part, len(set(part.tolist()))
    sizes = np.bincount(part)
    for p in np.flatnonzero(sizes == 1):
        if sizes[p] != 1:                  # already grew from an earlier merge
            continue
        v = int(np.flatnonzero(part == p)[0])
        nbrs = list(children[v]) + ([int(parent[v])] if parent[v] != -1 else [])
        target = min((part[u] for u in nbrs), key=lambda q: sizes[q])
        part[v] = target
        sizes[target] += 1
        sizes[p] = 0
    ids, part = np.unique(part, return_inverse=True)
    return part.astype(np.int64), len(ids)


def node_groups(edge_index, n_nodes, n_parts=10, max_overshoot=0.6):
    """Subtree id (0..n_parts-1) per node, plus the number of subtrees built."""
    parent, children, root = build_tree(edge_index, n_nodes)
    return partition_subtrees(parent, children, root, n_parts, max_overshoot)


# ----------------------------------------------------------------------
# self-test with ape-style numbering (tips 0..n-1, root n, internal preorder)
# ----------------------------------------------------------------------
def _ape_like_tree(newick_children, n_tips):
    """Build a symmetric edge_index from a nested-tuple tree."""
    edges, next_internal = [], [n_tips]
    tip_counter = [0]

    def walk(node):
        if node is None:
            t = tip_counter[0]; tip_counter[0] += 1
            return t
        me = next_internal[0]; next_internal[0] += 1
        for child in node:
            c = walk(child)
            edges.append((me, c))
        return me

    walk(newick_children)
    src = np.array([e[0] for e in edges]); dst = np.array([e[1] for e in edges])
    return np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])


def _balanced(depth):
    return None if depth == 0 else (_balanced(depth - 1), _balanced(depth - 1))


def _random_tree(n_tips, rng):
    nodes = [None] * n_tips
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        b = nodes.pop(j); a = nodes.pop(i)
        nodes.insert(i, (a, b))
    return nodes[0]


def _connected(part, parent):
    """Each subtree has exactly one node whose parent is outside it."""
    tops = [v for v in range(len(part))
            if parent[v] == -1 or part[parent[v]] != part[v]]
    return len(tops) == len(set(part.tolist()))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name, tree, n_tips in [("balanced 128 tips", _balanced(7), 128),
                               ("random 200 tips", _random_tree(200, rng), 200),
                               ("random 50 tips", _random_tree(50, rng), 50)]:
        ei = _ape_like_tree(tree, n_tips)
        n = 2 * n_tips - 1
        parent, children, root = build_tree(ei, n)
        part, k = partition_subtrees(parent, children, root, 10)
        sizes = np.bincount(part, minlength=10)
        assert sizes.sum() == n and (part >= 0).all() and _connected(part, parent)
        assert (sizes[:k] > 1).all()
        print(f"{name}: n={n}, subtrees={k}, sizes={sizes.tolist()}")