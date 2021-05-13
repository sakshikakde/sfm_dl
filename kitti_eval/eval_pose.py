# For my trained model..
# python3 eval_pose.py  --pred_dir ../../sfmLearner_checkpoints/outputs/pose_noSSIM

# For pretrained model..
# python3 eval_pose.py  --pred_dir ../../sfmLearner_checkpoints/outputs/pose_pretrained

from __future__ import division
import os
import numpy as np
import argparse
from glob import glob
from pose_evaluation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--gtruth_dir", default='./pose_eval_data/pose_data/ground_truth/09/', type=str, 
    help='Path to the directory with ground-truth trajectories')
parser.add_argument("--pred_dir", type=str, 
    help="Path to the directory with predicted trajectories")
args = parser.parse_args()

def main():
    pred_files = glob(args.pred_dir + '/*.txt')
    print("length of pred_files", len(pred_files))
    ate_all = []
    for i in range(len(pred_files)):
        gtruth_file = args.gtruth_dir + os.path.basename(pred_files[i])
        if not os.path.exists(gtruth_file):
            continue
        ate = compute_ate(gtruth_file, pred_files[i])
        
        if ate == False:
            continue
        ate_all.append(ate)
    ate_all = np.array(ate_all)
    print("Results:",ate_all.shape)
    print("Predictions dir: %s" % args.pred_dir)
    print("ATE mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))
main()