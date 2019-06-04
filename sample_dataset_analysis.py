#!/usr/bin/env python
"""
A simple example of how to use the ProtRank package.

File sample_dataset.dat contains synthetic data: randomly
generated (lognormally distributed) counts for 2000 proteins
measured under 8 different conditions. 20% of all counts are
zero. In the analysis, A-conditions are assumed as reference,
B-conditions are assumed as treated.
The first ten proteins have systematically lower (or higher)
counts in the B conditions as compared with the A conditions
(differential abundance analysis). ProtRank correctly
identifies all of them and adds one more protein (P480) which
is a false positive.
"""


import ProtRank


# define, which columns are compared (this can be a subset of all columns contained in the data)
what_to_compare = [[['A1', 'B1'], ['A2', 'B2'], ['A3', 'B3'], ['A4', 'B4']]]
description = 'A_vs_B_sample_dataset'

# function load_data returns the data structure used by ProtRank in the analysis
dataset = ProtRank.load_data('sample_dataset.dat')

# inspect some basic properties of the loaded dataset
ProtRank.data_stats(dataset, what_to_compare)

# search for differentially abundant proteins in the data
significant_proteins = ProtRank.rank_proteins(dataset, what_to_compare, description)

# plot the logarithmic fold change values of the found proteins
ProtRank.plot_lfc(dataset, what_to_compare, description, significant_proteins, 'A vs. B')
