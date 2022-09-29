import pickle
import numpy as np
from scipy import stats
import argparse

num_sets = 10

def case_to_key(dset, dset_idx, st_pos):
    if dset == "eth":
        if dset_idx == 0:
            if st_pos[0] == 5:
                return 1
            else:
                return 0
        elif dset_idx == 1:
            if st_pos[0] == 2:
                return 2
            else:
                return 3
        else:
            raise Exception("Dataset doesn't exist")
    elif dset == "ucy":
        if dset_idx == 0:
            if st_pos[0] == 7.5:
                return 5
            else:
                return 4
        elif dset_idx == 1:
            if st_pos[0] == 7.5:
                return 7
            else:
                return 6
        elif dset_idx == 2:
            if st_pos[0] == 7.5:
                return 9
            else:
                return 8
        else:
            raise Exception("Dataset doesn't exist")
    else:
        raise Exception("Dataset doesn't exist")

def organize_rst(filename):
    with open(filename, "rb") as fp:
        raw_results = pickle.load(fp)

    success_tally = []
    min_ped_dists = []
    path_lengths = []
    path_irregularity = []
    for i in range(num_sets):
        success_tally.append([])
        min_ped_dists.append([])
        path_lengths.append([])
        path_irregularity.append([])

    for r_result in raw_results:
        case = r_result[0]
        result = r_result[1]
        case_num = case_to_key(case[0], case[1], case[2])
        if case_num >= num_sets:
            continue
        if not (result[0] == 0):
            success_tally[case_num].append(1)
            min_ped_dists[case_num].append(result[1])
            path_lengths[case_num].append(result[3])
            path_irregularity[case_num].append(result[4])
        else:
            success_tally[case_num].append(0)
            min_ped_dists[case_num].append(-1)
            path_lengths[case_num].append(-1)
            path_irregularity[case_num].append(-1)

    return (success_tally, min_ped_dists, path_lengths, path_irregularity)

def cust_mean_std(array):
    gd_elems = []
    for i, e in enumerate(array):
        if (not e == -1):
            gd_elems.append(e)
    if len(gd_elems) == 0:
        return 0, 0
    else:
        gd_elems = np.array(gd_elems)
        return round(np.mean(gd_elems), 3), round(np.std(gd_elems), 3)

def ind_evaluate(raw_results):

    success_tally, min_ped_dists, path_lengths, path_irregularity = raw_results

    success_rates = [0] * num_sets
    for i in range(num_sets):
        if not (len(success_tally[i]) == 0):
            success_rates[i] = round(np.sum(success_tally[i]) / len(success_tally[i]), 4)
            #success_rates[i] = np.std(success_tally[i])

    min_ped_dists_mean = [0] * num_sets
    min_ped_dists_std = [0] * num_sets
    path_lengths_mean = [0] * num_sets
    path_lengths_std = [0] * num_sets
    path_irregularity_mean = [0] * num_sets
    path_irregularity_std = [0] * num_sets

    for i in range(num_sets):
        min_ped_dists_mean[i], min_ped_dists_std[i] = cust_mean_std(min_ped_dists[i])
        path_lengths_mean[i], path_lengths_std[i] = cust_mean_std(path_lengths[i])
        path_irregularity_mean[i], path_irregularity_std[i] = cust_mean_std(path_irregularity[i])

    return (np.array(success_rates), 
            np.array(min_ped_dists_mean), 
            np.array(min_ped_dists_std), 
            np.array(path_lengths_mean), 
            np.array(path_lengths_std), 
            np.array(path_irregularity_mean),
            np.array(path_irregularity_std))

def refine(raw_results, combined_tally):
    success_tally, min_ped_dists, path_lengths, path_irregularity = raw_results

    for i in range(num_sets):
        for j, ind in enumerate(combined_tally[i]):
            if ind == 0:
                success_tally[i][j] = 0
                min_ped_dists[i][j] = -1
                path_lengths[i][j] = -1
                path_irregularity[i][j] = -1

    return (success_tally, min_ped_dists, path_lengths, path_irregularity)

def delete_negative(array):
    new_array = []
    for elem in array:
        if not (elem == -1):
            new_array.append(elem)
    return new_array

if __name__ == "__main__":

    x = input("Reactive Agents? (y/n): ")
    if (x == 'y'):
        react_flag = True
    else:
        react_flag = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=int)
    parser.add_argument('--policy1', type=int)
    parser.add_argument('--policy2', type=int)
    args = parser.parse_args()

    if not ((args.metric == 0) or (args.metric == 1) or (args.metric == 2)):
        raise Exception('Metric can only be 0, 1 or 2!')
    if not ((args.policy1 >= 0) and (args.policy1 <= 5) and
            (args.policy2 >= 0) and (args.policy2 <= 5)):
        raise Exception('Policy number can only be 0, 1, 2, 3, 4 or 5!')

    if react_flag:
        exp_names = ["ped_nopred_react", "ped_linear_react", "ped_sgan_react",
                     "group_nopred_react", "group_auto_react", "group_auto_laser_react"] 
    else:
        exp_names = ["ped_nopred", "ped_linear", "ped_sgan",
                     "group_nopred", "group_auto", "group_auto_laser"] 

    directory = "results/"
    rst_dict = {}
    total_tally = []
    all_results = []
    for exp in exp_names:
        fname = directory + exp + ".txt"
        raw_results = organize_rst(fname)
        all_results.append(raw_results)
        results = ind_evaluate(raw_results)
        total_tally.append(raw_results[0])
        rst_dict[exp] = results
        print("====================", exp, "====================")
        print(results[0])
        print(results[1])
        print(results[3])

    print("================================================")
    print("================================================")
    print("================================================")

    metric = args.metric
    p_threshold = 0.05
    num_exp = len(all_results)
    set1 = [args.policy1]
    set2 = [args.policy2]
    for i in set1:
        for j in set2:
            data1 = all_results[i][metric]
            data2 = all_results[j][metric]
            p_values = []
            for k in range(num_sets):
                data_set1 = delete_negative(data1[k])
                data_set2 = delete_negative(data2[k])
                try:
                    cp_rst = stats.mannwhitneyu(data_set1, data_set2, alternative="two-sided")
                    p_values.append(round(cp_rst.pvalue, 4))
                except ValueError:
                    p_values.append(np.inf)
            print(exp_names[i] + " VS " + exp_names[j])
            print("Flow: ", p_values[::2])
            print("Cross: ", p_values[1::2])
            print("Flow (p<"+str(p_threshold)+"?): ", np.array(p_values[::2]) < p_threshold)
            print("Cross (p<"+str(p_threshold)+"?): ", np.array(p_values[1::2]) < p_threshold)
            print("==============================================")

    with open("final_results.txt", "wb") as fp:
        pickle.dump(rst_dict, fp)

















