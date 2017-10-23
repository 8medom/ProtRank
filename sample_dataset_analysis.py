#!/usr/bin/env python
"""
A simple example of how to use the ProtRank package.

File sample_dataset.dat is artificially created. It contains
randomly generated (lognormally distributed) counts for 2000
proteins measured under 8 different conditions. 20% of all
counts are zero. In the analysis, A-conditions are assumed
as reference, B-conditions are assumed as treated.
"""


import ProtRank


# here we define, which columns are compared
what_to_compare = [[['A1', 'B1'], ['A2', 'B2'], ['A3', 'B3'], ['A4', 'B4']]]

# here we used the load_data function which returns the data structure used by ProtRank
my_data = ProtRank.load_data('sample_dataset.dat')

# here we search for the significant proteins in the data
significant_proteins = ProtRank.rank_proteins(my_data, what_to_compare, 'sample_dataset', num_bootstrap_realizations = 100, prior_count = 1)
