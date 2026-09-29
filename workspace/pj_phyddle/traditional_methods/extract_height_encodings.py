#!/usr/bin/env python3


# Code to obtain CHV embeddings for simulated data.
# Builds on existing code to obtain CBLV+S encodings in standard Phyddle utilities.
# Edits made by Kate Truman with AI assistance

import glob
import os
import csv
import numpy as np
from dendropy import Tree
import re
NUMPY_FLOAT_FMT_STR = '{{:0.{:d}e}}'.format(16)
def ndarray_to_flat_str(x):
    """Converts a numpy.ndarray into flattend csv vector."""

    # numpy formatter for floats & ints
    def numpy_formatter(y):
        if y % 1 == 0:
            return "{:d}".format(int(y))
        else:
            return NUMPY_FLOAT_FMT_STR.format(y)

    # convert ndarray to formatted string
    s = np.array2string(x, separator=',', max_line_width=1e200,
                        threshold=1e200, edgeitems=1e200,
                        floatmode='maxprec',
                        formatter={'float_kind': numpy_formatter})
    # remove brackets, whitespace
    s = re.sub(r'[\[\]\n ]', '', string=s)
    # endline
    # s = s + '\n'
    return s


INPUT_PATTERN = "/mnt/DiversificationGraphInference/thirty_thousand_trees/simulate/out.*.downsampled.tre"
OUTPUT_FILE = "chv.csv"

def encode_chv_heights(phy, tree_width=1999, tree_encode_type='height_brlen', rescale=True):
    """
    Encode Compact Height-reordered Vector 

    # num columns equals tree_width, 0-padding
    # returns tensor with following rows
    # 0:  internal node root-distance
    # 1:  leaf node branch length
    # 2:  internal node branch length

    Arguments:
        phy (dendropy.Tree):     phylogenetic tree
        tree_width (int):        number of columns (max. num. taxa)
                                 in CPVS array
        tree_encode_type (str):  type of tree encoding ('height_only' or
                                 'height_brlen')
        rescale:                 set tree height to 1 then encode, if True

    Returns:
        numpy.ndarray: The encoded CDV+S tensor.
    """

    # data dimensions
    num_tree_col = 0
    if tree_encode_type == 'height_only':
        num_tree_col = 1
    elif tree_encode_type == 'height_brlen':
        num_tree_col = 3

    # initialize workspace
    phy.calc_node_root_distances(return_leaf_distances_only=False)
    heights    = np.zeros( (tree_width, num_tree_col) )
    height_idx = 0

    # postorder traversal to rotate nodes by clade-length
    for nd in phy.postorder_node_iter():
        if nd.is_leaf():
            nd.treelen = 0.
        else:
            children           = nd.child_nodes()
            ch_treelen         = [ (ch.edge.length + ch.treelen) for ch in children ]
            nd.treelen         = sum(ch_treelen)
            ch_treelen_rank    = np.argsort( ch_treelen )[::-1]
            children_reordered = [ children[i] for i in ch_treelen_rank ]
            nd.set_children(children_reordered)

    # inorder traversal to fill matrix
    phy.seed_node.edge.length = 0
    for nd in phy.inorder_node_iter():

        if nd.is_leaf():
            if tree_encode_type == 'height_brlen':
                heights[height_idx,1] = nd.edge.length
                height_idx += 1
        else:
            heights[height_idx,0] = nd.root_distance
            if tree_encode_type == 'height_brlen':
                heights[height_idx,2] = nd.edge.length
            height_idx += 1

    # stack the phylo and states tensors
    if rescale:
        heights = heights / np.max(heights)
    phylo_tensor = np.hstack( [heights] )
    return phylo_tensor

# Find all tree files
files = glob.glob(INPUT_PATTERN)

# Sort numerically by NUM
files.sort(key=lambda x: int(
    os.path.basename(x).split(".")[1]
))

print(f"Found {len(files)} tree files")


with open(OUTPUT_FILE, "w", newline="") as outfile:

    writer = csv.writer(outfile)

    first_row = True

    for i, filename in enumerate(files, 1):

        basename = os.path.basename(filename)

        # Extract NUM from:
        # out.NUM.downsampled.tre
        num = basename.split(".")[1]

        if i % 100 == 0:
            print(f"[{i}/{len(files)}] Processing {basename}")

        # Read tree into a DendroPy phylo object
        tree = Tree.get(
            path=filename,
            schema="newick"
        )
        # Encode the phylogeny
        cpvs_data = encode_chv_heights(tree)

        # Convert flattened array to comma-separated string
        encoded = ndarray_to_flat_str(cpvs_data.flatten())

        # Split into individual CSV columns
        values = encoded.strip().split(",")
        if i % 100 == 0:
            print(len(values))

        # Write header based on first result
        if first_row:
            header = ["NUM"] + [
                f"chv_{j}" for j in range(len(values))
            ]
            writer.writerow(header)
            first_row = False

        # Write NUM followed by encoding
        writer.writerow([num] + values)


print(f"\nFinished.")
print(f"Output written to: {OUTPUT_FILE}")
