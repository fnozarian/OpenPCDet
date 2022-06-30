#!/usr/bin/python3
# python calc_mean_mAP.py \
# --exp_paths output/cfgs/da-nuscenes-kitti_models/secondiou/secondiou_old_anchor/default/eval/eval_all_default/nusc_kitti_secondiou_old_anchor_trial1/ \
#     output/cfgs/da-nuscenes-kitti_models/secondiou/secondiou_old_anchor/default/eval/eval_all_default/nusc_kitti_secondiou_old_anchor_trial2/

import argparse
import os
import re

import numpy as np



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_paths', required=True, nargs=argparse.REMAINDER,
                        help='--exp_paths <eval_path-1>, <eval_path-2> ..')
    parser.add_argument('--log_tb', action='store_true', default=True, help='')
    args = parser.parse_args()
    return args


def get_sorted_text_files(dirpath):
    a = [s for s in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, s)) and s.endswith('.txt')]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a



def calc_mean_mAP():
    """
    Takes n experiments and calculate mean of max class mAP
    """
    args = parse_config()
    assert args.exp_paths is not None
    exp_paths = [str(x) for x in args.exp_paths]    


    metric = ["Car AP_R40@0.70, 0.70, 0.70"]
    pattern = re.compile(r'({0})'.format('|'.join(metric)))
    max_results=[]
    eval_list=None

    print("\n#--------------------Calculate Mean mAP-----------------------#\n")
    print("\nDefined Metric")
    for m in metric:
        print(m)
    print("\nExperiment(s))")
    for e in exp_paths:
        print(e)
    
    

    all_eval_results = []
    for curr_res_dir in exp_paths:
        if not os.path.isdir(curr_res_dir): 
            continue
        curr_eval_list_file = os.path.join(curr_res_dir, "eval_list_val.txt")
        if eval_list is None and os.path.isfile(curr_eval_list_file):
            with open(curr_eval_list_file) as f_eval:
                eval_list = list(set(map(int, f_eval.readlines())))# take only unique entries 
                print("\nEvaluated Epochs")
                print(*[str(i) for i in eval_list], sep=",")

        
    

        text_files = get_sorted_text_files(curr_res_dir)
        if len(text_files)==0:
            print("No text file found containing results")
            continue
        
        # get data from file 
        eval_results=[]

        for file_ in text_files:# traverse all file to find evaluation results
            selected_file=os.path.join(curr_res_dir, file_)
            print("\nScanning {} for evaluated results\n".format(selected_file))# can be filtered based on date-time

            
            
            line_numbers=[]
            linenum = 0
            res_=[]
            with open(selected_file) as fp:
                for line in fp:
                    linenum += 1
                    if pattern.search(line) != None: # If a match is found 
                        line_numbers.append(linenum+2) # BEV
                        line_numbers.append(linenum+3) # 3D
                    
                    if linenum in line_numbers:
                        if "bev" in line:
                            res_.append(np.fromstring( line.strip().split("bev  AP:")[1], dtype=np.float64, sep=',' ))
                        if "3d" in line:
                            res_.append(np.fromstring( line.strip().split("3d   AP:")[1], dtype=np.float64, sep=',' ))
                        
                    if len(res_)==2:
                        eval_results.append(np.array(res_).flatten())
                        res_=[]
        
        # reshape records based on eval_list
        eval_results=np.array(eval_results).reshape(len(eval_list),-1)
        all_eval_results.append(eval_results)
        print("\nmAP(s)")
        print(*[str(np.round_(i, decimals=2)) for i in eval_results], sep="\n")


        current_max=np.max(eval_results, axis=0)
        max_results.append(current_max)
        print("\nMax mAP")
        print(*[str(np.round_(i, decimals=2)) for i in current_max], sep=", ")
        print("\nBest Epoch (with moderate scroe): " + str(eval_list[np.where(max_results[0][4] == eval_results[:,4])[0][0]]))
        

    print("\n\n----------------Final Results----------------\n\n")
    # all results have been added
    max_results=np.array(max_results)
    print("Max mAP(s)\n")
    print(*[str(np.round_(i, decimals=2)) for i in max_results], sep="\n")
    

    mean_res=np.mean(max_results, axis=0)
    print("\nMean mAP")
    print(*[str(np.round_(i, decimals=2)) for i in mean_res], sep=", ")

    if args.log_tb:
        from tensorboardX import SummaryWriter

        re_trial = re.compile(r'trial(\d)')
        trials = re_trial.findall(" ".join(exp_paths))
        trials = sorted(map(int, trials))
        trial_splits = re_trial.split(exp_paths[0])
        new_trial = "trial{0}-{1}".format(str(trials[0]), str(trials[-1]))
        new_exp = "".join([trial_splits[0], new_trial, trial_splits[-1]])
        new_exp_dir = os.path.join(new_exp, "tensorboard_val")
        all_eval_results = np.dstack(all_eval_results)
        mean_eval_results = np.mean(all_eval_results, -1)
        #print(mean_eval_results)
        classes = ['Car_bev','Car_3d']
        difficulties = ['easy_R40', 'moderate_R40', 'hard_R40']
        num_diffs = len(difficulties)
        num_classes = len(classes)
        class_wise_mean_eval_results = mean_eval_results.reshape((-1, num_diffs, num_classes), order='F')

        tb_log = SummaryWriter(log_dir=new_exp_dir)
        for i, cls in enumerate(classes):
            for j, diff in enumerate(difficulties):
                key = cls + "/" + diff
                for k, step in enumerate(eval_list):
                    val = class_wise_mean_eval_results[k, j, i]
                    tb_log.add_scalar(key, val, step)

if __name__ == "__main__":
    calc_mean_mAP()
