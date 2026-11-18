"""
part_newick
===========
Write one tree as a Newick string in which every branch carries the id of
the subtree (part) that node_groups assigned to it.

The branch above a node gets that node's part id. The root has no branch
above it but is still labeled with its own part id.

Example:
    from phyddle.subtree_partition import node_groups
    from part_newick import part_newick, to_nexus

    part_np, sizes = node_groups(edge_index.numpy(), x_np.shape[0], n_parts)
    nwk = part_newick(edge_index, part_np, num_nodes=x_np.shape[0])
    print(nwk)
    with open(f"tree_{g}_parts.tre", "w") as f:
        f.write(to_nexus(nwk))      # opens in FigTree, colour by "part"
"""
import numpy as np


def _to_numpy(a):
    if a is None:
        return None
    if hasattr(a, "detach"):          # torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _find_root(ei, adj, n):
    """Directed edges: the node with no incoming edge.
    Undirected edges: the only internal node of degree 2 (rooted binary tree)."""
    pairs = set(map(tuple, ei.T.tolist()))
    directed = all((v, u) not in pairs for (u, v) in pairs if u != v)
    if directed:
        has_parent = np.zeros(n, bool)
        has_parent[ei[1]] = True
        roots = np.flatnonzero(~has_parent)
    else:
        roots = np.array([u for u in range(n) if len(adj[u]) == 2])
    if len(roots) != 1:
        raise ValueError(f"Could not identify a unique root (candidates: "
                         f"{roots.tolist()}). Pass root=<node index>.")
    return int(roots[0])


def part_newick(edge_index, part, num_nodes=None, root=None,
                branch_lengths=None, tip_names=None, style="figtree"):
    """
    Args:
        edge_index:     [2, E] edges for ONE tree, local node numbering 0..n-1.
                        May be directed (parent->child) or undirected.
        part:           [n] subtree id per node (output of node_groups).
        num_nodes:      number of nodes (defaults to len(part)).
        root:           root node index; inferred if None.
        branch_lengths: optional [n] length of the branch above each node.
        tip_names:      optional sequence of names indexed by node.
        style:          "figtree" -> name[&part=3]:0.1   (FigTree / ggtree)
                        "name"    -> name|p3:0.1         (any Newick reader)
    Returns:
        Newick string ending in ';'.
    """
    ei = _to_numpy(edge_index).astype(int)
    part = _to_numpy(part).astype(int)
    bl = _to_numpy(branch_lengths)
    n = int(num_nodes) if num_nodes is not None else part.shape[0]

    adj = [set() for _ in range(n)]
    for u, v in ei.T:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    adj = [sorted(a) for a in adj]

    if root is None:
        root = _find_root(ei, adj, n)

    # breadth-first pass from the root gives parent -> children
    children = [[] for _ in range(n)]
    seen = np.zeros(n, bool)
    seen[root] = True
    order = [root]
    for u in order:                    # list grows while iterating
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                children[u].append(v)
                order.append(v)
    if not seen.all():
        raise ValueError("Tree is disconnected: some nodes unreachable "
                         "from the root.")

    # build strings bottom-up (iterative, so deep trees don't hit recursion limits)
    s = [None] * n
    for u in reversed(order):
        is_tip = not children[u]
        name = (str(tip_names[u]) if tip_names is not None
                else (f"n{u}" if is_tip else ""))
        if style == "figtree":
            label = f"{name}[&part={part[u]}]"
        elif style == "name":
            label = f"{name or f'i{u}'}|p{part[u]}"
        else:
            raise ValueError("style must be 'figtree' or 'name'")
        head = "" if is_tip else "(" + ",".join(s[c] for c in children[u]) + ")"
        length = f":{bl[u]:g}" if (bl is not None and u != root) else ""
        s[u] = head + label + length
    return s[root] + ";"


def to_nexus(newick, name="tree1"):
    """Wrap a Newick string in a minimal NEXUS block (FigTree, ggtree::read.beast)."""
    return f"#NEXUS\nbegin trees;\n\ttree {name} = [&R] {newick}\nend;\n"