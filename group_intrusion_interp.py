import numpy as np
import pickle
from scipy import stats
import argparse

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


def interp_rst(fname):
    intru_dir = "group_intrusion_rst/"
    with open(intru_dir + fname, 'rb') as f:
        data = pickle.load(f)

    num_scenes = 10
    rst = [[], [], [], [], [], [], [], [], [], []]
    for d in data:
        idx = case_to_key(d[0][0], d[0][1], d[0][2])
        if (np.sum(np.array(d[1]))) > 0:
            rst[idx].append(0)
        else:
            rst[idx].append(1)

    data_dir = "results/"
    with open(data_dir + fname, 'rb') as f:
        data = pickle.load(f)
    for d in data:
        idx = case_to_key(d[0][0], d[0][1], d[0][2])
        if (d[1][0] == 0):
            rst[idx].append(0)

    for i in range(num_scenes):
        print(round(np.mean(np.array(rst[i])) * 100, 2), end=' ')
    print()
    return rst
            
x = input("Reactive Agents? (y/n): ")
if (x == 'y'):
    react_flag = True
else:
    react_flag = False

parser = argparse.ArgumentParser()
parser.add_argument('--policy1', type=int)
parser.add_argument('--policy2', type=int)
args = parser.parse_args()

if not ((args.policy1 >= 0) and (args.policy1 <= 5) and
            (args.policy2 >= 0) and (args.policy2 <= 5)):
        raise Exception('Policy number can only be 0, 1, 2, 3, 4 or 5!')

if not react_flag:
    experiments = ["ped_nopred.txt",
                   "ped_linear.txt",
                   "ped_sgan.txt",
                   "group_nopred.txt",
                   "group_auto.txt",
                   "group_auto_laser.txt"]
else:
    experiments = ["ped_nopred_react.txt",
                   "ped_linear_react.txt",
                   "ped_sgan_react.txt",
                   "group_nopred_react.txt",
                   "group_auto_react.txt",
                   "group_auto_laser_react.txt"]

all_results = []
for exp in experiments:
    print(exp)
    rst = interp_rst(exp)
    all_results.append(rst)

print("=================================================")

p_threshold = 0.05
set1 = [args.policy1]
set2 = [args.policy2]
num_sets = 10
for i in set1:
    for j in set2:
        data1 = all_results[i]
        data2 = all_results[j]
        p_values = []
        for k in range(num_sets):
            if k == 0:
                all_data1 = data1[k]
                all_data2 = data2[k]
            else:
                all_data1 += data1[k]
                all_data2 += data2[k]
            cp_rst = stats.mannwhitneyu(data1[k], data2[k], alternative="two-sided")
            #cp_rst = stats.ttest_ind(data_set1, data_set2)
            p_values.append(round(cp_rst.pvalue, 4))
        print(experiments[i] + " VS " + experiments[j])
        print("Flow: ", p_values[::2])
        print("Cross: ", p_values[1::2])
        print("Flow (p<"+str(p_threshold)+"?): ", np.array(p_values[::2]) < p_threshold)
        print("Cross (p<"+str(p_threshold)+"?): ", np.array(p_values[1::2]) < p_threshold)
        cp_rst = stats.mannwhitneyu(all_data1, all_data2, alternative="two-sided")
        print(round(cp_rst.pvalue, 4))
        print("==============================================")

