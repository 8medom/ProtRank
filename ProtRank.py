"""
A module for differential abundance analysis of proteomics data based
on assigning a ranking score to the proteins by the logarithmic fold
changes of their counts as well as their counts becoming zero.

matus.medo@unifr.ch, 2017
"""


import sys
import numpy as np                                          # for numerics
import pandas as pd                                         # for data frames
import csv                                                  # CSV File Reading and Writing
import scipy.stats.mstats as mstats                         # to rank proteins whilst ignoring NaNs
from numpy.random import seed, randint                      # to proteinrate bootstrap data
import matplotlib.pyplot as plt                             # for figure plotting
import seaborn as sns                                       # for nicer plotting
sns.set_style('whitegrid')


def load_data(file_name):
    """
    Loads the input data with protein counts, protein names, and condition labels.
    The column separator is determined automatically.
    The first line is assumed to contain the condition labels.
    Each of all subsequent lines is assumed to correspond to one protein whose
    name is in the first column, and whose counts in all conditions are
    in the following columns.

    Parameters
    ----------
    file_name: name of the input file (or the full path to the input file)

    Returns
    -------
    out : Pandas DataFrame
    Data frame with protein counts indexed by protein names and columns given by
    the condition labels read from the input file.
    """
    with open(file_name, 'r') as f:
        header_line = f.readline()
        sniffer = csv.Sniffer()
        separator = sniffer.sniff(header_line).delimiter    # automatically determine the column separator
        condition_labels = header_line.strip().split(separator)[1:]
        proteins = []
        for line in f:
            proteins.append(line.split(separator)[0])
    raw_data = np.genfromtxt(file_name, delimiter = separator, skip_header = 1, usecols = range(1, len(condition_labels) + 1))
    raw_data[np.isnan(raw_data)] = 0                        # replace NaNs with zeros
    print 'loaded input data with {} proteins that have been measured under {} different conditions'.format(raw_data.shape[0], raw_data.shape[1])
    return pd.DataFrame(raw_data, index = proteins, columns = condition_labels)


def data_stats(data, what_to_compare):
    """
    Does basic analysis of the provided data frame.
    
    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    what_to_compare : nested (two-levels) list
    A nested list with pairwise comparisons of columns (described by column labels).
    """
    print '\nmeasurement data contain results for {} proteins and {} different conditions'.format(data.shape[0], len(data.columns))
    print 'list of measured conditions: {}'.format(', '.join(data.columns))

    where_NZ = np.nonzero(data.values)
    num_NZ = where_NZ[0].size
    num_Z = data.values.size - num_NZ
    print 'there are {} zero counts ({:.1f}% of all)'.format(num_Z, 100 * float(num_Z) / data.values.flatten().size)

    mean_count = np.mean(data.values[where_NZ])
    median_count = np.median(data.values[where_NZ])
    print 'median peptide count is {:.2e} (computed over non-zero entries only)'.format(median_count)
    print 'mean protein count is {:.2e} (computed over non-zero entries only)'.format(mean_count)

    min_count = np.min(data.values[where_NZ])
    max_count = np.max(data.values)
    print 'smallest non-zero protein count is {:.0f}; largest protein count is {:.2e} (ratio {:.2e})'.format(min_count, max_count, float(max_count) / min_count)
    
    comparisons, comparisons_subset, which_group = identify_columns(data, what_to_compare)
    comparisons_flat = np.array(sum(sum(comparisons, []), []))      # construct a 1D array with all the columns included in the analysis
    analyzed_data = data.values[:, comparisons_flat]                # data subset where only the columns involved in the required comparisons are included
    count_all, count_05, count_10, count_20, count_40 = 0, 0, 0, 0, 0
    for p in range(analyzed_data.shape[0]):
        for comp in range(which_group.size / 2):
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
    print '\npresence of irregular zeros:'
    print 'out of {} comparisons with one zero and one non-zero count, the non-zero count exceeds 0.5 * median in {:.1f}% of all cases'.format(count_all, 100 * float(count_05) / count_all)
    print 'out of {} comparisons with one zero and one non-zero count, the non-zero count exceeds 1.0 * median in {:.1f}% of all cases'.format(count_all, 100 * float(count_10) / count_all)
    print 'out of {} comparisons with one zero and one non-zero count, the non-zero count exceeds 2.0 * median in {:.1f}% of all cases'.format(count_all, 100 * float(count_20) / count_all)
    print 'out of {} comparisons with one zero and one non-zero count, the non-zero count exceeds 4.0 * median in {:.1f}% of all cases'.format(count_all, 100 * float(count_40) / count_all)
    print '(the smaller these fractions, the smaller is the problem with irregular zeros in the data)'


def rank_proteins(data, what_to_compare, description = 'ProtRank_analysis', prior_count = 1, num_bootstrap_realizations = 100, FDR_threshold = 0.15):
    """
    Rank proteins by their differential expression.
    
    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    what_to_compare : nested (two-levels) list
    A nested list with pairwise comparisons of columns (described by column labels).
    description : string, optional
    A suffix for the output files (useful to distinguish results from various analyses). The default is 'ProtRank_analysis'.
    prior_count : int, optional
    This count is added to all actual protein counts when the logarithmic fold changes are computed. It attenuates the impact of proteins with low counts. The default is 1.
    num_bootstrap_realizations : int, optional
    The number of bootstrap realizations that are used to estimate the proteins' false detection rate (FDR). The default is 100.
    FDR_threshold : float, optional
    False detection rate below which proteins are reported as differentially expressed. The default is 0.15.
    """
    num_groups = len(what_to_compare)
    num_comparisons = sum([len(group) for group in what_to_compare])
    print '\nthere are {} groups of comparisons provided, they contain the following number of pairs: [{}]'.format(num_groups, ', '.join([str(len(group)) for group in what_to_compare]))
    print 'in total, {} comparisons will be included in the analysis'.format(num_comparisons)

    comparisons, comparisons_subset, which_group = identify_columns(data, what_to_compare)

    comparisons_flat = np.array(sum(sum(comparisons, []), []))      # construct a 1D array with all the columns included in the analysis
    if len(comparisons_flat) <> len(set(comparisons_flat)):         # current bootstrap implementation is not adequate for a case where a single column is used multiple times (e.g., A vs. B and A vs. C)
        print 'beware: current bootstrap implementation treats all compared columns as independent which is not adequate here because one or more columns are used multiple times'
        print 'in practice, this means that the provided FDR estimates are likely to be overly optimistic (i.e., lower than they should be)'
    analyzed_data = data.values[:, comparisons_flat]                # data subset where only the columns involved in the required comparisons are included
    best_score, signs_of_best_score = get_best_scores(analyzed_data, prior_count, comparisons_subset, which_group)
    sorted_real_score = -np.sort(-best_score)                       # twice minus to sort from the highest to the lowest

    seed(0)                                                         # analysis of bootstrap data
    data_flat = analyzed_data.flatten()                             # auxiliary 1D array to create the bootstrapped data
    bootstrap_scores, bootstrap_scores2, num_better = np.zeros(analyzed_data.shape[0]), np.zeros(analyzed_data.shape[0]), np.zeros(analyzed_data.shape[0])
    print '{} bootstrap realizations:'.format(num_bootstrap_realizations),
    for n in range(num_bootstrap_realizations):
        print n + 1,
        sys.stdout.flush()
        bootstrap_data = np.zeros_like(analyzed_data)               # initialize the bootstrapped data array
        bootstrapped_data = data_flat[randint(data_flat.size, size = data_flat.size)].reshape((analyzed_data.shape[0], analyzed_data.shape[1]))
        boot_score, signs_of_boot_score = get_best_scores(bootstrapped_data, prior_count, comparisons_subset, which_group)
        for g in range(data.shape[0]):
            num_better[g] += np.where(boot_score >= sorted_real_score[g])[0].size
        sorted_score = -np.sort(-boot_score)

        #raw_input()
        bootstrap_scores += sorted_score
        bootstrap_scores2 += np.power(sorted_score, 2)
    bootstrap_scores /= num_bootstrap_realizations
    bootstrap_scores2 /= num_bootstrap_realizations
    num_better /= num_bootstrap_realizations
    save_bootstrap_profile(data, bootstrap_scores, bootstrap_scores2, description)
    significant_proteins = save_results(data, best_score, signs_of_best_score, num_better, comparisons, description, num_bootstrap_realizations, FDR_threshold)
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
            if len(pair) <> 2:
                print 'error: comparison {} does not involve two columns'.format(pair)
                sys.exit(1)
            try: col_ref = column_labels.index(pair[0])
            except:
                print 'error: in the data, there is no column labeled {}'.format(pair[0])
                sys.exit(1)
            try: col_comp = column_labels.index(pair[1])
            except:
                print 'error: in the data, there is no column labeled {}'.format(pair[1])
                sys.exit(1)
            comparisons[-1].append([col_ref, col_comp])
            comparisons_subset[-1].append([count_columns, count_columns + 1])
            which_group[count_comparisons] = n                      # to which group which comparison belongs
            count_comparisons += 1
            count_columns += 2
    return comparisons, comparisons_subset, which_group


def evaluate_one_comparison(data, prior_count, pair):
    """
    Internal function: Evaluates the rank score of all proteins in one comparison.
    The rank score is small for proteins with large positive logarithmic fold change values and proteins whose count changes from zero to a non-zero value.
    The rank score is close to one for proteins with large negative logarithmic fold change values and proteins whose count changes from a non-zero value to zero.
    
    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    prior_count : int, optional
    This count is added to all actual protein counts when the logarithmic fold changes are computed. It attenuates the impact of proteins with low counts. The default is 2.
    pair : list with two integers
    The integers define the columns involved in the comparison.
    The first element is the reference column, the second element is the column with a modified condition.
    
    Returns
    -------
    out : ndarray with the rank score of all proteins
    """

    # compute the fraction of VV, VZ, ZV, and ZZ pairs for the given pair of columns
    VV_proteins = np.where((data[:, pair[0]]  > 0) & (data[:, pair[1]]  > 0) == True)[0]
    xVV = VV_proteins.size / float(data.shape[0])
    VZ_proteins = np.where((data[:, pair[0]]  > 0) & (data[:, pair[1]] == 0) == True)[0]
    xVZ = VZ_proteins.size / float(data.shape[0])
    ZV_proteins = np.where((data[:, pair[0]] == 0) & (data[:, pair[1]]  > 0) == True)[0]
    xZV = ZV_proteins.size / float(data.shape[0])
    ZZ_proteins = np.where((data[:, pair[0]] == 0) & (data[:, pair[1]] == 0) == True)[0]
    xZZ = ZZ_proteins.size / float(data.shape[0])
    score_vector = np.zeros(data.shape[0])
    score_vector[:] = -1
    lfc_vals = np.zeros(data.shape[0])
    #ZV_nz_vals = np.zeros(data.shape[0])                           # all commented lines in this function can be used to distinguish ZV and VZ proteins by the magnitude of their nonzero count
    #VZ_nz_vals = np.zeros(data.shape[0])
    for row in range(data.shape[0]):
        if data[row, pair[1]] > 0 and data[row, pair[0]] > 0:
            lfc_vals[row] = np.log2((prior_count + data[row, pair[1]]) / (prior_count + data[row, pair[0]]))
        else:
            lfc_vals[row] = np.nan
        #if data[row, pair[1]] > 0 and data[row, pair[0]] == 0:
            #ZV_nz_vals[row] = data[row, pair[1]]
        #if data[row, pair[1]] == 0 and data[row, pair[0]] > 0:
            #VZ_nz_vals[row] = data[row, pair[0]]
    rank_lfc = mstats.rankdata(np.ma.masked_invalid(-lfc_vals))     # the largest lfc values get the smallest rank; inf and nan entries get zero rank
    #rank_ZV = mstats.rankdata(-ZV_nz_vals)
    #rank_VZ = mstats.rankdata(-VZ_nz_vals)
    #for pep in VV_proteins:
        #score_vector[pep] = (rank_lfc[pep] - 0.5) / VV_proteins.size
    score_vector[VV_proteins] = (rank_lfc[VV_proteins] - 0.5) / VV_proteins.size
    score_vector[VZ_proteins] = 1 - 0.5 * xVZ
    score_vector[ZV_proteins] = 0.5 * xZV
    score_vector[ZZ_proteins] = -1                                     # by setting negative score here, we mark that this is a ZZ pair
    return score_vector


def get_best_scores(data, prior_count, comparisons, which_group):
    """
    Internal function: Searches for a combination of directions (looking at a ranking from the top or bottom, respectively) that maximizes the aggregate rank score of each protein.
    
    Parameters
    ----------
    data : Pandas DataFrame
    Data frame with protein counts, protein names, and condition labels.
    prior_count : int, optional
    This count is added to all actual protein counts when the logarithmic fold changes are computed. It attenuates the impact of proteins with low counts. The default is 2.
    comparisons: nested (two-levels) list
    A nested list with pairwise comparisons of columns.
    which_group: list
    A list with group affiliation for each comparion from comparisons. This is needed because all comparisons that belong to the same group are viewed in the same way (from the top or bottom, respectively).

    Returns
    -------
    r : ndarray with the best aggregate rank score for each protein
    s : 2D ndarray with the views that lead to the best rank score for each protein
    """
    num_groups = len(comparisons)
    num_comparisons = sum([len(group) for group in comparisons])    # determine how many comparisons are there to be made in total
    score = np.ones((data.shape[0], num_comparisons))               # initialize the score vectors for all comparisons
    counter = 0                                                     # compute the rank score of all proteins for each individual comparison
    for group in comparisons:
        for pair in group:
            score[:, counter] = evaluate_one_comparison(data, prior_count, pair)
            counter += 1

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
        for p in range(data.shape[0]):                       # for the current sign configuration, compute the score of all peptides
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


def save_results(data, score_vector, signs_of_best_score, num_better, comparisons, description, num_bootstrap_realizations, FDR_threshold):
    """
    Internal function: Saves the list of all proteins ranked by the their aggregate rank score together with their FDR estimates.
    """
    sort_order = np.argsort(-score_vector)
    o = open('prs-{}.dat'.format(description), 'w')     # save protein rank scores (prs)
    for n in range(len(comparisons)):
        o.write('# comparisons in group {}:'.format(n + 1))
        for pair in comparisons[n]:
            o.write(' {}-{}'.format(pair[0], pair[1]))
        o.write('\n')

    o.write('# FDR values computed using {} bootstrap realizations of the data\n'.format(num_bootstrap_realizations))
    o.write('# rank\trow\tprotein\trank score\tFDR\tsings in groups\n')
    FDR = 0
    significant_proteins = []
    chosen_rows = []
    for n in range(data.shape[0]):
        row = sort_order[n]
        sign_strings = []
        for c in range(signs_of_best_score[row,:].size):
            if signs_of_best_score[row, c] == 0: sign_strings.append('+')
            else: sign_strings.append('-')
        FDR_prev = FDR
        FDR = num_better[n] / (n + 1)
        if FDR < FDR_prev: FDR = FDR_prev
        if FDR > 1: FDR = 1
        if FDR <= FDR_threshold:
            chosen_rows.append(row)
            significant_proteins.append(data.index[row])
        if score_vector[row] < 1e-2:
            o.write('{}\t{}\t{}\t{:.4e}\t{:.3f}\t{}\n'.format(n + 1, sort_order[n], data.index[row], score_vector[row], FDR, ''.join(sign_strings)))
        else:
            o.write('{}\t{}\t{}\t{:.6f}\t{:.3f}\t{}\n'.format(n + 1, sort_order[n], data.index[row], score_vector[row], FDR, ''.join(sign_strings)))
    o.close()

    o = open('prs-significant-{}.dat'.format(description), 'w')
    o.write('# list of {} proteins with FDR <= {} (FDR values computed using {} bootstrap realizations of the data)\n'.format(len(significant_proteins), FDR_threshold, num_bootstrap_realizations))
    for protein in significant_proteins:
        o.write('{}\n'.format(protein))
    o.close()

    print '\nthere are {} proteins that have FDR {} or less'.format(len(significant_proteins), FDR_threshold)
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


def plot_logcounts_for(data, what_to_compare, list_of_proteins, description = 'ProtRank_analysis', prior_count = 1):
    num_proteins = len(list_of_proteins)
    comparisons, comparisons_subset, which_group = identify_columns(data, what_to_compare)
    comparisons_flat = np.array(sum(sum(comparisons, []), []))      # construct a 1D array with all the columns included in the analysis

    fig = plt.figure(figsize = (4.5, 0.7 * num_proteins))
    for n, protein in enumerate(list_of_proteins):
        protein_counts = data.loc[protein]
        x, positions, values, colors = 0, [], [], []
        for group in comparisons:
            for pair in group:
                positions.append(x)
                values.append(np.log2(prior_count + protein_counts[pair[0]]))
                positions.append(x + 1)
                values.append(np.log2(prior_count + protein_counts[pair[1]]))
                if protein_counts[pair[0]] < protein_counts[pair[1]]:     # color for upregulated proteins
                    colors.extend([(0.302, 0.686, 0.290), (0.302, 0.686, 0.290)])
                elif protein_counts[pair[0]] > protein_counts[pair[1]]:   # color for downregulated proteins
                    colors.extend([(0.894, 0.102, 0.110), (0.894, 0.102, 0.110)])
                else: colors.extend(['0.7', '0.7'])                 # color for proteins that do not change their count
                x += 2.4
            x += 0.6

        ax = fig.add_axes([0.1, 0.9 - 0.8 * n / num_proteins, 0.8, 0.5 / num_proteins])
        ax.bar(positions, values, color = colors)
        ax.set_xlim([-0.2, positions[-1] + 1.0])
        ax.set_xticks([])
        ax.set_ylim(ymin = 0)
        if n == len(list_of_proteins) - 1:
            for m, col in enumerate(comparisons_flat):
                ax.text(0.4 + positions[m], -0.05 * plt.gca().get_ylim()[1], list(data.columns)[col], ha = 'center', va = 'top', fontsize = 10)
        if num_proteins % 2 == 1 and n == (num_proteins - 1) / 2:
            ax.text(-0.5, 0.5 * plt.gca().get_ylim()[1], r'log_2 (n_0 + count)', rotation = 90, ha = 'right', va = 'center', fontsize = 14)
        elif num_proteins % 2 == 0 and n == num_proteins / 2:
            ax.text(-0.5, plt.gca().get_ylim()[1], r'log_2 (n_0 + count)', rotation = 90, ha = 'right', va = 'center', fontsize = 14)
        ax.locator_params(axis = 'y', nbins = 3)
        ax.tick_params(axis = 'y', pad = 2)
        ax.text(1, 1, protein, transform = plt.gca().transAxes, ha = 'right', va = 'bottom', fontsize = 10)
    plt.savefig('significant_proteins-logcounts-{}.pdf'.format(description), bbox_inches = 'tight', padinches = 0.02)
    plt.close()
