"""
A Python module for differential abundance analysis of proteomics data
based on assigning a ranking score to the proteins by the logarithmic
fold changes of their counts as well as their counts becoming zero.

matus.medo@unifr.ch, 2019
"""

from __future__ import division             # for simultaneous support of Python 2 and 3
from __future__ import print_function       # for simultaneous support of Python 2 and 3
from __future__ import unicode_literals     # for simultaneous support of Python 2 and 3

from builtins import str                    # for simultaneous support of Python 2 and 3
from builtins import range                  # for simultaneous support of Python 2 and 3
from past.utils import old_div              # for simultaneous support of Python 2 and 3

import sys                                  # system-level functions
import numpy as np                          # for numerics
from numpy.random import seed, randint      # for the bootstrap analysis
import pandas as pd                         # for data frames
pd.options.display.width = 160              # to increase the width of the printed text to 160 characters (default is 80)
import scipy.stats.mstats as mstats         # for ranking the proteins by their log-fold-change whilst ignoring NaNs
import matplotlib as mpl
import matplotlib.pyplot as plt             # for figure plotting
plt.style.use('seaborn-whitegrid')          # for nicer plotting
from matplotlib.patches import Rectangle
try:                                        # colorcet blue-red colormap is preferred
    import colorcet as cc
    cmap = cc.m_coolwarm_r
except:                                     # fall-back option: matplotlib's red-blue colormap
    from matplotlib import cm
    cmap = cm.RdBu


median_lfc_window_size = 100                # sliding window size (in terms of the number of proteins) used to estimated median absolute lfc


def load_data(file_name, separator = '\t', ignore_cols = [], index_col = 0, comments = '#'):
    """
    Loads the input data with protein counts, protein names, and condition labels.
    The first line is assumed to contain the condition labels.
    Each of all subsequent lines is assumed to correspond to one protein whose
    name is in the first column, and whose counts in all conditions are
    in the following columns.

    Parameters
    ----------
    file_name: string
    Name of the input file (or the full path to the input file).
    separator: char, optional
    Character used to separate the data columns. The default is tab ('\t').
    ignore_cols: list of strings
    Which columns should be ignored. The default is an empty list.
    index_col: int
    Which column contains the index. The default is the first column (column zero).

    Returns
    -------
    out : Pandas DataFrame
    Data frame with protein counts indexed by protein names and columns given by
    the condition labels read from the input file.
    """
    raw_data = pd.read_csv(file_name, comment = comments, delimiter = separator, index_col = index_col)
    if len(ignore_cols) != 0:           # remove the columns that should be ignored
        raw_data = raw_data.drop(columns = ignore_cols)
    raw_data = raw_data.fillna(0)       # replace NaNs with zeros
    print('\nProtRank.load_data():\nloaded input data with {} proteins that have been measured under {} different conditions'.format(raw_data.shape[0], raw_data.shape[1]))
    print('list of the measured conditions: {}'.format(', '.join(raw_data.columns)))
    where_NZ = np.nonzero(raw_data.values)
    num_NZ = where_NZ[0].size
    num_Z = raw_data.values.size - num_NZ
    print('in the data, {:.1f}% of all counts are zeros'.format(100. * num_Z / raw_data.values.flatten().size))
    return raw_data


def data_stats(data, what_to_compare = None, ignore_missed = True):
    """
    Does basic analysis of the provided data frame.

    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    what_to_compare : nested (two-levels) list
    A nested list with pairwise comparisons of columns (described by column labels).
    """
    if what_to_compare == None:
        print('\nProtRank.data_stats() for all data columns:')
        median_count = np.median(data.values[where_NZ])
        print('median count is {:.2e} (computed over non-zero entries only)'.format(median_count))
        min_count = np.min(data.values[where_NZ])
        max_count = np.max(data.values[where_NZ])
        perc10, perc90 = np.percentile(data.values[where_NZ], [10, 90])
        print('ratio between the largest and the smallest non-zero count is {:.2e}'.format(float(max_count) / min_count))
        print('ratio between the 90th and the 10th percentile non-zero count is {:.2e}'.format(float(perc90) / perc10))
    else:
        print('\nProtRank.data_stats() for the provided columns:')
        comparisons, comparisons_subset, which_groups = identify_columns(data, what_to_compare)
        comparisons_flat = np.array(sum(sum(comparisons, []), []))  # construct a 1D array with all the columns included in the analysis
        print('{} comparisons provided: {}'.format(len(comparisons_flat), what_to_compare))
        keep = []
        for one_row in data.values:
            if np.where(one_row[comparisons_flat] > 0)[0].size > 0: keep.append(True)
            else: keep.append(False)
        if ignore_missed:                                           # filter out the rows with only zero counts
            analyzed_data = data.values[keep, :][:, comparisons_flat]
            print('after ignoring {} rows with only zero counts, {} rows remain'.format(data.values.shape[0] - sum(keep), analyzed_data.shape[0]))
        else:
            analyzed_data = data.values[:, comparisons_flat]        # keep also the rows with only zero counts
            print('{} rows with only zero counts will be included in the analysis; it might be better to remove ignore_missed = False to prevent that'.format(data.values.shape[0] - sum(keep)))
        print('in the analyzed data, {:.1f}% of all counts are zeros'.format(100. * np.where(analyzed_data == 0)[0].size / analyzed_data.size))
        where_NZ = np.nonzero(analyzed_data)
        median_count = np.median(analyzed_data[where_NZ])
        print('median count is {:.2e} (computed over non-zero entries only)'.format(median_count))

        min_count = np.min(analyzed_data[where_NZ])
        max_count = np.max(analyzed_data[where_NZ])
        perc10, perc90 = np.percentile(analyzed_data[where_NZ], [10, 90])
        print('ratio between the largest and the smallest non-zero count is {:.2e}'.format(float(max_count) / min_count))
        print('ratio between the 90th and the 10th percentile non-zero count is {:.2e}'.format(float(perc90) / perc10))

        count_all, count_05, count_10, count_20, count_40 = 0, 0, 0, 0, 0
        for p in range(analyzed_data.shape[0]):
            for comp in range(old_div(len(comparisons_flat), 2)):
                if analyzed_data[p, 2 * comp] == 0 and analyzed_data[p, 2 * comp + 1] > 0:
                    count_all += 1
                    if analyzed_data[p, 2 * comp + 1] > 0.5 * median_count: count_05 += 1
                    if analyzed_data[p, 2 * comp + 1] > median_count: count_10 += 1
                    if analyzed_data[p, 2 * comp + 1] > 2 * median_count: count_20 += 1
                    if analyzed_data[p, 2 * comp + 1] > 4 * median_count: count_40 += 1
                if analyzed_data[p, 2 * comp] > 0 and analyzed_data[p, 2 * comp + 1] == 0:
                    count_all += 1
                    if analyzed_data[p, 2 * comp] > 0.5 * median_count: count_05 += 1
                    if analyzed_data[p, 2 * comp] > median_count: count_10 += 1
                    if analyzed_data[p, 2 * comp] > 2 * median_count: count_20 += 1
                    if analyzed_data[p, 2 * comp] > 4 * median_count: count_40 += 1
        num_all = old_div(analyzed_data.shape[0] * len(comparisons_flat), 2)
        print('statistics of irregular missing values:')
        print('  in total, there are {} comparisons involving a zero and a non-zero value ({:.1f}% of all)'.format(count_all, 100. * count_all / num_all))
        #~for [frac, count_here] in [[0.5, count_05], [1, count_10], [2, count_20], [4, count_40]]:
        for [frac, count_here] in [[1, count_10]]:
            print('  out of {} comparisons, {} involve a zero value and a non-zero exceeding {:.1f} * median ({:.1f}% of all)'.format(num_all, count_here, frac, 100. * count_here / num_all))
        print('  (the higher the fraction, the bigger the problem with irregular zeros in the data)')


def rank_proteins(data, what_to_compare, description = 'ProtRank_analysis', prior_count = 1, num_bootstrap_realizations = 100, FDR_threshold = 0.1, rel_rank_ZV = 0.1, ignore_missed = True, column_normalization = True, save_pvals = False):
    """
    Rank proteins by their differential abundance.

    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    what_to_compare : nested (two-levels) list
    A nested list with pairwise comparisons of columns (described by column labels).
    description : string, optional
    A suffix for the output files (useful to distinguish results from various analyses). The default is 'ProtRank_analysis'.
    prior_count : float, optional
    This count is added to all actual protein counts when the logarithmic fold changes are computed. It attenuates the impact of proteins with low counts. The default is 1.
    num_bootstrap_realizations : int, optional
    The number of bootstrap realizations that are used to estimate the proteins' false detection rate (FDR). The default is 100.
    FDR_threshold : float, optional
    False detection rate below which proteins are reported as differentially abundant. The default is 0.1.
    ignore_missed: boolean, optional
    Whether to ignore the rows without positive counts. The default is True.
    column_normalization: boolean, optional
    Whether to apply column normalization on the observed counts. The default is True.
    save_pvals: boolean, optional
    Whether to save bootstrap estimates of differential expression p-values in the output file. The default is False.

    Returns
    -------
    significant_proteins: list
    List of the proteins whose FDR is below a given threshold in the differential expression analysis.
    """
    num_groups = len(what_to_compare)
    num_comparisons = sum([len(group) for group in what_to_compare])
    print('\nProtRank.rank_proteins():\nthere are {} groups of comparisons provided, they contain the following number of pairs: [{}]'.format(num_groups, ', '.join([str(len(group)) for group in what_to_compare])))
    print('in total, {} comparisons will be included in the analysis'.format(num_comparisons))

    comparisons, comparisons_subset, which_group = identify_columns(data, what_to_compare)

    comparisons_flat = np.array(sum(sum(comparisons, []), []))      # construct a 1D array with all the columns included in the analysis
    if ignore_missed:
        keep = []
        for one_row in data.values:
            if np.where(one_row[comparisons_flat] > 0)[0].size > 0: keep.append(True)
            else: keep.append(False)
        analyzed_data = data.values[keep, :][:, comparisons_flat].astype(float)
        analyzed_data_pd = data[keep]
        print('for the given comparisons, {} rows with at least one nonzero count remain'.format(analyzed_data.shape[0]))
    else:
        analyzed_data = data.values[:, comparisons_flat].astype(float)  # data subset where only the columns involved in the required comparisons are included
        print('keeping all {} rows, including {} rows with only zero counts'.format(data.values.shape[0], data.values.shape[0] - sum(keep)))
    if column_normalization:                                            # apply column normalization on all columns (mainly for the logarithmic fold change estimation; it does not impact the ranking results, except for the prior_count relative value)
        col_sums = np.sum(analyzed_data, axis = 0)
        avg_col_sum = np.mean(col_sums)
        for col in range(col_sums.size):
            analyzed_data[:, col] *= avg_col_sum / col_sums[col]
        print('applied column normalization factors: {}'.format(', '.join(['{:.2f}'.format(val) for val in avg_col_sum / col_sums])))
    best_score, signs_of_best_score, all_lfc_values = get_best_scores(analyzed_data, prior_count, comparisons_subset, which_group, rel_rank_ZV, return_lfc = True)
    sorted_real_score = -np.sort(-best_score)                           # twice minus to sort from the highest to the lowest

    seed(0)                                                             # analysis of bootstrap data
    data_flat = analyzed_data.flatten()                                 # auxiliary 1D array to create the bootstrapped data
    bootstrap_scores, bootstrap_scores2, num_better, pvals = np.zeros(analyzed_data.shape[0]), np.zeros(analyzed_data.shape[0]), np.zeros(analyzed_data.shape[0]), np.zeros(analyzed_data.shape[0])
    print('{} bootstrap realizations:'.format(num_bootstrap_realizations), end=' ')
    for n in range(num_bootstrap_realizations):
        print(n + 1, end=' ')
        sys.stdout.flush()
        bootstrapped_data = np.zeros_like(analyzed_data)                # initialize the bootstrapped data array
        for comp in range(old_div(analyzed_data.shape[1], 2)):          # randomization by comparison (always two columns are reshuffled together)
            random_order = randint(analyzed_data.shape[0], size = analyzed_data.shape[0])
            bootstrapped_data[:, 2 * comp] = analyzed_data[random_order, 2 * comp]
            bootstrapped_data[:, 2 * comp + 1] = analyzed_data[random_order, 2 * comp + 1]
        boot_score, signs_of_boot_score = get_best_scores(bootstrapped_data, prior_count, comparisons_subset, which_group, rel_rank_ZV)
        for g in range(analyzed_data.shape[0]):
            num_better[g] += np.where(boot_score >= sorted_real_score[g])[0].size
            if boot_score[g] > sorted_real_score[g]: pvals[g] += 1
        sorted_score = -np.sort(-boot_score)
        bootstrap_scores += sorted_score
        bootstrap_scores2 += np.power(sorted_score, 2)
    bootstrap_scores /= num_bootstrap_realizations
    bootstrap_scores2 /= num_bootstrap_realizations
    num_better /= num_bootstrap_realizations
    pvals /= num_bootstrap_realizations
    #~save_bootstrap_profile(analyzed_data, bootstrap_scores, bootstrap_scores2, description)
    significant_proteins = save_results(analyzed_data_pd, best_score, signs_of_best_score, num_better, pvals, what_to_compare, description, num_bootstrap_realizations, FDR_threshold, save_pvals, all_lfc_values, prior_count)
    return significant_proteins


def identify_columns(data, what_to_compare):
    """
    Internal function: Identify the column IDs that correspond to a given nested list with column labels.

    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    what_to_compare : nested (two-levels) list
    A nested list with pairwise comparisons of columns (described by column labels).

    Returns
    -------
    comparisons : a nested list with corresponding column IDs
    comparisons_subset : a nested list with column IDs in the subset of analyzed data (hence these column IDs progressively increase from 0 to the number of analyzed columns minus 1
    which_group: ndarray which stores to which group each comparison belongs
    """
    if not type(what_to_compare[0][0]) == list: # the provided list of comparisons does not have the expected depth, warn the user
        print('\nERROR: provided list of comparisons is expected to be grouped')
        print('add one level of brackets to your list of comparisons to resolved this')
        print('ProtRank will then search for DE proteins that change in the same direction in all individual comparisons')
        sys.exit(1)

    num_comparisons = sum([len(group) for group in what_to_compare])    # determine how many comparisons are there to be made in total
    comparisons = []                                                    # which columns are to be included in comparisons
    comparisons_subset = []                                             # the same as above, but on the subset including only the analyzed columns
    which_group = np.zeros(num_comparisons, dtype = int)
    count_comparisons, count_columns = 0, 0
    column_labels = list(data.columns)
    for n, group in enumerate(what_to_compare):
        comparisons.append([])
        comparisons_subset.append([])
        for pair in group:
            if len(pair) != 2:
                print('\nERROR: comparison {} does not involve two columns'.format(pair))
                sys.exit(1)
            try: col_ref = column_labels.index(pair[0])
            except:
                print('\nERROR: in the data, there is no column labeled {}'.format(pair[0]))
                sys.exit(1)
            try: col_comp = column_labels.index(pair[1])
            except:
                print('\nERROR: in the data, there is no column labeled {}'.format(pair[1]))
                sys.exit(1)
            comparisons[-1].append([col_ref, col_comp])
            comparisons_subset[-1].append([count_columns, count_columns + 1])
            which_group[count_comparisons] = n                      # to which group which comparison belongs
            count_comparisons += 1
            count_columns += 2
    return comparisons, comparisons_subset, which_group


def evaluate_one_comparison(data, prior_count, pair, rel_rank_ZV):
    """
    Internal function: Evaluates the rank score of all proteins in one comparison.
    The rank score is small for proteins with large positive logarithmic fold change values and proteins whose count changes from zero to a non-zero value.
    The rank score is close to one for proteins with large negative logarithmic fold change values and proteins whose count changes from a non-zero value to zero.

    Parameters
    ----------
    data : ndarray
    NumPy array with the analyzed protein counts.
    prior_count : float, optional
    This count is added to all actual protein counts when the logarithmic fold changes are computed. It attenuates the impact of proteins with low counts. The default is 2.
    pair : list with two integers
    The integers define the columns involved in the comparison.
    The first element is the reference column, the second element is the column with a modified condition.

    Returns
    -------
    s : ndarray with the rank score of all proteins (for the given comparison)
    lfc : ndarray with the logarithmic fold change for all proteins (for the given comparison)
    """

    # compute the fraction of VV, VZ, ZV, and ZZ pairs for the given pair of columns
    VV_proteins = np.where((data[:, pair[0]]  > 0) & (data[:, pair[1]]  > 0) == True)[0]
    #xVV = VV_proteins.size / float(data.shape[0])
    VZ_proteins = np.where((data[:, pair[0]]  > 0) & (data[:, pair[1]] == 0) == True)[0]
    #xVZ = VZ_proteins.size / float(data.shape[0])
    ZV_proteins = np.where((data[:, pair[0]] == 0) & (data[:, pair[1]]  > 0) == True)[0]
    #xZV = ZV_proteins.size / float(data.shape[0])
    ZZ_proteins = np.where((data[:, pair[0]] == 0) & (data[:, pair[1]] == 0) == True)[0]
    #xZZ = ZZ_proteins.size / float(data.shape[0])
    score_vector = np.zeros(data.shape[0])
    score_vector[:] = -1
    lfc_vals = np.zeros(data.shape[0])
    for row in range(data.shape[0]):
        if data[row, pair[1]] > 0 and data[row, pair[0]] > 0:
            lfc_vals[row] = np.log2(float(prior_count + data[row, pair[1]]) / (prior_count + data[row, pair[0]]))
        else:
            lfc_vals[row] = np.nan
    rank_lfc = mstats.rankdata(np.ma.masked_invalid(-lfc_vals))     # the largest lfc values get the smallest rank; inf and nan entries get zero rank
    score_vector[VV_proteins] = (rank_lfc[VV_proteins] - 0.5) / VV_proteins.size
    score_vector[ZV_proteins] = rel_rank_ZV
    score_vector[VZ_proteins] = 1 - rel_rank_ZV
    score_vector[ZZ_proteins] = -1                                  # by setting negative score here, we mark that this is a ZZ pair
    return score_vector, lfc_vals


def evaluate_best(data, comparisons, score, which_group):
    num_groups = len(comparisons)
    num_comparisons = sum([len(group) for group in comparisons])    # determine how many comparisons are there to be made in total
    best_score = np.zeros(data.shape[0])                # initialize the best score vector
    signs_of_best_score = np.empty((data.shape[0], num_groups), dtype = np.int8)
    sign_counter = 0
    signs = np.zeros(num_groups, dtype = np.int8)
    while sign_counter < 2**num_groups:                 # go over all 2^num_groups permutations of change sign for all groups
        to_convert = sign_counter                       # to_convert is a counter that is converted in its binary representation - we thus get all possible sign configurations from all + to all -
        for n in range(num_groups):
            if to_convert % 2 == 1: signs[n] = 1        # sign 1 is for looking at the ranking from the bottom (largest negative lfc and VZs)
            else: signs[n] = 0                          # sign 0 is for looking at the ranking from the top (largest positive lfc and ZVs)
            to_convert /= 2
        for p in range(data.shape[0]):                  # for the current sign configuration, compute the score of all peptides
            prod = 1
            for n in range(num_comparisons):
                if score[p, n] < 0:
                    pass                                # negative score marks that this was a comparison with two zero values, which has no impact on the total rank score
                else:
                    if signs[which_group[n]] == 0: prod *= -np.log(score[p, n]) # we look at the ranking from the top
                    else: prod *= -np.log(1 - score[p, n])                      # we look at the ranking from the bottom
            if prod > best_score[p]:                    # if the result is better than for this peptide all previous sign configurations, store it
                best_score[p] = prod
                signs_of_best_score[p, :] = signs
        sign_counter += 1                               # go to the next sign configuration
    return best_score, signs_of_best_score


def get_best_scores(data, prior_count, comparisons, which_group, rel_rank_ZV, return_lfc = False):
    """
    Internal function: Searches for a combination of directions (looking at a ranking from the top or bottom, respectively) that maximizes the aggregate rank score of each protein.

    Parameters
    ----------
    data : ndarray
    NumPy array with the analyzed protein counts.
    prior_count : float, optional
    This count is added to all actual protein counts when the logarithmic fold changes are computed. It attenuates the impact of proteins with low counts. The default is 2.
    comparisons: nested (two-levels) list
    A nested list with pairwise comparisons of columns.
    which_group: list
    A list with group affiliation for each comparion from comparisons. This is needed because all comparisons that belong to the same group are viewed in the same way (from the top or bottom, respectively).
    return_lfc: boolean, optional
    If True, the underlying lfc values are returned.

    Returns
    -------
    r : ndarray with the best aggregate rank score for each protein
    s : 2D ndarray with the views that lead to the best rank score for each protein
    lfc: 2D ndarray with the logarithmic fold change values for all proteins and all comparisons
    """
    num_groups = len(comparisons)
    num_comparisons = sum([len(group) for group in comparisons])    # determine how many comparisons are there to be made in total
    score = np.ones((data.shape[0], num_comparisons))               # initialize the score vectors for all comparisons
    lfc = np.ones((data.shape[0], num_comparisons))                 # initialize the score vectors for all comparisons
    counter = 0                                                     # compute the rank score of all proteins for each individual comparison
    for group in comparisons:
        for pair in group:
            score[:, counter], lfc[:, counter] = evaluate_one_comparison(data, prior_count, pair, rel_rank_ZV)
            counter += 1
    best_score, signs_of_best_score = evaluate_best(data, comparisons, score, which_group)
    if return_lfc: return best_score, signs_of_best_score, lfc
    else: return best_score, signs_of_best_score


def save_results(data, score_vector, signs_of_best_score, num_better, pvals, what_to_compare, description, num_bootstrap_realizations, FDR_threshold, save_pvals, all_lfc_values, prior_count):
    """
    Internal function: Saves the list of all proteins ranked by the their aggregate rank score
    together with their FDR estimates and LFC values.
    """
    comparisons, comparisons_subset, which_groups = identify_columns(data, what_to_compare)
    num_comparisons = 0
    for group in what_to_compare: num_comparisons += len(group)
    o = open('prs-{}.dat'.format(description), 'w')     # save ProtRank scores (prs)
    o.write('# ProtRank analysis results of a dataset with {} rows and {} comparisons\n'.format(data.shape[0], num_comparisons))
    for n in range(len(what_to_compare)):
        o.write('# comparisons in group {}: '.format(n + 1))
        o.write(' & '.join('{} vs. {}'.format(pair[0], pair[1]) for pair in what_to_compare[n]))
        o.write('\n')
    o.write('# FDR values computed using {} bootstrap realizations of the data\n'.format(num_bootstrap_realizations))
    if save_pvals: o.write('# rank\trow\trank score\tp-val\tFDR\tsigns in groups\n')
    else: o.write('# rank\trow\tLFC\trank score\tFDR\tsigns in groups\n')

    sort_order = np.argsort(-score_vector)  # to iterate over proteins from the highest rank score to the lowest
    lfc_sorted = all_lfc_values[sort_order]

    FDR = 0
    significant_proteins = []
    chosen_rows = []
    shift = old_div(median_lfc_window_size, 2)
    lfc_vals = np.zeros(len(what_to_compare))
    for n in range(data.shape[0]):
        if n < shift:
            window_start, window_end = 0, median_lfc_window_size
        elif n > data.shape[0] - shift:
            window_start, window_end = data.shape[0] - median_lfc_window_size, data.shape[0]
        else: window_start, window_end = n - shift, n + shift
        values_for_median = np.abs(lfc_sorted[window_start:window_end,:])
        if np.all(np.isnan(values_for_median)): median_abs_lfc = 0
        else: median_abs_lfc = np.nanmedian(values_for_median)      # median absolute lfc around this rank position
        row = sort_order[n]
        pos = 0
        for g, group in enumerate(what_to_compare):
            lfc_vals[g] = 0
            for pair in group:
                if data[pair[1]].iloc[row] > 0 and data[pair[0]].iloc[row] > 0:
                    lfc_vals[g] += lfc_sorted[n, pos]
                elif data[pair[1]].iloc[row] > 0 and data[pair[0]].iloc[row] == 0:
                    lfc_vals[g] += median_abs_lfc
                elif data[pair[1]].iloc[row] == 0 and data[pair[0]].iloc[row] > 0:
                    lfc_vals[g] += -median_abs_lfc
                pos += 1
            lfc_vals[g] /= len(group)
        lfc_string = ' '.join(['{:.2f}'.format(x) for x in lfc_vals])
        #raw_input()
        sign_strings = []
        for c in range(signs_of_best_score[row,:].size):
            if signs_of_best_score[row, c] == 0: sign_strings.append('+')
            else: sign_strings.append('-')
        FDR_prev = FDR
        FDR = old_div(num_better[n], (n + 1))
        if FDR < FDR_prev: FDR = FDR_prev
        if FDR > 1: FDR = 1
        if FDR <= FDR_threshold:
            chosen_rows.append(row)
            significant_proteins.append(data.index[row])
        o.write('{}\t{}\t{}\t{:.6f}\t'.format(n + 1, data.index[row], lfc_string, np.power(score_vector[row], 1. / num_comparisons)))
        if save_pvals: o.write('{}\t'.format(pvals[n]))
        o.write('{:.5f}\t{}\n'.format(FDR, ''.join(sign_strings)))
    o.close()

    o = open('prs-significant-{}.dat'.format(description), 'w')
    o.write('# list of {} proteins with FDR <= {} (FDR values computed using {} bootstrap realizations of the data)\n'.format(len(significant_proteins), FDR_threshold, num_bootstrap_realizations))
    for protein in significant_proteins:
        o.write('{}\n'.format(protein))
    o.close()
    print('\n{}: there are {} proteins that have FDR {} or less'.format(description, len(significant_proteins), FDR_threshold))
    return significant_proteins


def save_bootstrap_profile(data, score, score2, description):
    """
    Save the list of all peptides/proteins ranked by the their rank score.
    """
    score2 -= np.power(score, 2)
    score2[np.where(score2 < 0)[0]] = 0
    o = open('prs-bootstrap-{}.dat'.format(description), 'w')
    o.write('# rank\tE(score)\tstd(score)\n')
    for n in range(data.shape[0]):
        o.write('{}\t{:.4e}\t{:.4e}\n'.format(n + 1, score[n], np.sqrt(score2[n])))
    o.close()


def plot_lfc(data, what_to_compare, list_of_rows, description, group_labels = None, row_names = None, val_range = None, prior_count = 1., group_separation = 0.2, fig_title = None, title_font_size = 20, group_title_font_size = 16, id_font_size = 16):
    """
    Visualize the logarithmic fold change values for the given rows.

    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    what_to_compare : nested (two-levels) list
    A nested list with pairwise comparisons of columns (described by column labels).
    description: string
    String that is appended to the name of the output graphics file lfc_values-*.pdf.
    list_of_rows: list
    A list of rows (identified by the DataFrame index) with proteins that should be included in the plot (typically the significantly changing ones).
    group_labels: list
    A list of strings that are used as a header for each group of comparisons in the figure.
    row_names: list, optional
    A list of strings that specify the label of each row in the figure. When none is provided, the DataFrame index is used instead.
    val_range: float, optional
    The scale that is used to transform the computed lfc values in colors. When none is provided, 90th percentile of the absolute computed lfc values is used instead.
    prior_count : float, optional
    This count is added to all actual protein counts when the logarithmic fold changes are computed. It attenuates the impact of proteins with low counts. The default is 1.
    group_separation: float, optional
    The horizontal space that is used to separate groups of comparisons.
    fit_title: string, optional
    The figure title.
    group_title_font_size: float, optional
    Size of the font that is used for group headers.
    id_font_size: float, optional
    Size of the font that is used to label each row in the figure.
    """

    print('\nProtRank.plot_lfc():')
    comparisons, comparisons_subset, which_group = identify_columns(data, what_to_compare)
    comparisons_flat = np.array(sum(sum(comparisons, []), []))      # construct a 1D array with all the columns included in the analysis
    n_proteins = len(list_of_rows)
    n_cols = old_div(len(comparisons_flat), 2)
    n_groups = len(what_to_compare)        

    lfc_vals, abs_lfc_vals = [], []
    for row in list_of_rows:
        lfc_vals.append([])
        for group in what_to_compare:
            for pair in group:
                lfc = np.log2(float(prior_count + data[pair[1]].loc[row]) / (prior_count + data[pair[0]].loc[row]))
                lfc_vals[-1].append(lfc)
                if data[pair[1]].loc[row] > 0 and data[pair[0]].loc[row] > 0: abs_lfc_vals.append(abs(lfc))
    if val_range == None:       # determine the value range automatically as the 90th percentile of the absolute lfc values
        val_range = np.percentile(abs_lfc_vals, 90)
        print('auto-determined range is {:.1f} (90th percentile of the displayed absolute lfc values)'.format(val_range))
    else:
        print('using the provided range {:.1f} (90th percentile of the displayed absolute lfc values)'.format(val_range, np.percentile(abs_lfc_vals, 90)))

    fig = plt.figure(figsize = (n_cols - 0.2 + (n_groups - 1) * group_separation, n_proteins - 0.2), frameon = False)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.6])
    for y, row in enumerate(list_of_rows):
        x = 0
        for group_number, group in enumerate(what_to_compare):
            for pair in group:
                color_index = int(127.5 + 127.5 * lfc_vals[y][x] / val_range)
                if color_index < 0: color_index = 0
                elif color_index > 255: color_index = 255
                xpos = x + group_separation * group_number
                ypos = n_proteins - 1 - y
                ax.add_patch(Rectangle((xpos, ypos), 0.8, 0.8, facecolor = cmap(color_index)))
                x += 1
        if row_names == None: name = row
        else: name = row_names[y]
        ax.text(-0.1, ypos + 0.35, name, ha = 'right', va = 'center', fontsize = id_font_size)    
    cbaxes = fig.add_axes([0.77, 0.7 - 0.6 * 3.8 / n_proteins, 0.04, 0.6 * 3.8 / n_proteins])
    norm = mpl.colors.Normalize(vmin = -val_range, vmax = val_range)
    if val_range >= 8: ticks = [-8, -4, 0, 4, 8]
    elif val_range >= 6: ticks = [-6, -3, 0, 3, 6]
    elif val_range >= 4: ticks = [-4, -2, 0, 2, 4]
    elif val_range >= 3: ticks = [-3, -2, -1, 0, 1, 2, 3]
    elif val_range >= 2: ticks = [-2, -1, 0, 1, 2]
    elif val_range >= 1: ticks = [-1, -0.5, 0, 0.5, 1]
    else: ticks = [-0.1, 0, 0.1]
    cb = mpl.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm, orientation = 'vertical', ticks = ticks, drawedges = False, label = 'LFC')
    cb.outline.set_linewidth(0)
    cb.ax.tick_params(labelsize = 12)
    ax.set_xlim([0, n_cols + (n_groups - 1) * group_separation - 0.2])
    ax.set_ylim([0, n_proteins - 0.2])
    if group_labels != None:
        if len(what_to_compare) == 1:                                   # there is just one group of comparisons and one string
            ax.text(0.5 * len(what_to_compare[0]) - 0.1, n_proteins - 0.1, group_labels, ha = 'center', va = 'bottom', fontsize = group_title_font_size)
        else:                                                           # the are multiple groups of comparisons, group_labels has to follow them
            if len(group_labels) != len(what_to_compare):
                print('warning: length of the list of provided group labels ({}) differs from the number of groups ({}) - no group headers will be displayed'.format(len(group_labels), len(what_to_compare)))
            else:
                x = 0
                for g in range(len(what_to_compare)):
                    ax.text(x + 0.5 * len(what_to_compare[g]) - 0.1, n_proteins - 0.1, group_labels[g], ha = 'center', va = 'bottom', fontsize = group_title_font_size)
                    x += len(what_to_compare[g]) + group_separation
    if fig_title != None:
        ax.text(0.5 * x - 0.1 - 0.5 * group_separation, n_proteins + 0.5, fig_title, ha = 'center', va = 'bottom', fontsize = title_font_size)
    ax.axis('off')
    fname = 'lfc_values-{}.pdf'.format(description)
    plt.savefig(fname, bbox_inches = 'tight', padinches = 0.02)
    plt.close()
    print('plot saved in file {}'.format(fname))


def load_significant(file_name, FDR_threshold = 0.1):
    """
    Loads signficant proteins from a ProtRank result file.
    This is a convenience function that helps to avoid repetitively running the significance analysis before plotting the results, for example.

    Parameters
    ----------
    file_name: string
    Name of a result file that has been produced by ProtRank.
    FDR_threshold : float, optional
    False detection rate below which proteins are reported as differentially abundant. The default is 0.1.

    Returns
    -------
    significant_proteins: list
    List of the proteins whose FDR is below a given threshold in the differential expression analysis.
    """
    try: f = open(file_name, 'r')
    except:
        print('input file {} not found'.format(file_name))
        return
    else:
        data = f.readlines()
        names_of_significant = []
        print('loading significant proteins from file {}:'.format(file_name))
        for n, line in enumerate(data):
            if line[0] == '#':
                if not line.startswith('# rank\trow\t'): print(line.strip())  # display the initial informative lines
            else:
                parts = line.split('\t')
                try: FDR_val = float(parts[-2])
                except:
                    print('\nERROR: failed to parse line "{}"'.format(line.strip()))
                    return
                else:
                    if FDR_val <= FDR_threshold: names_of_significant.append(parts[1])
                    else:   # we reached a non-significant protein, stop reading further
                        print('done - loaded {} significant proteins'.format(len(names_of_significant)))
                        return names_of_significant
