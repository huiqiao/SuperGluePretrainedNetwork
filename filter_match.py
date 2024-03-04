from pathlib import Path
import argparse
import random
import numpy as np
import pandas as pd

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, required=True,
        help='Path to the list of image pairs')
    parser.add_argument(
        '--match_dir', type=str, required=True,
        help='Path to the directory in which the .npz results')
    parser.add_argument(
        '--match_ratio_threshold', type=float, required=True,
        help='Math ratio threshold which we consider similar figures in ipc')
    parser.add_argument(
        '--similar_figure_file', type=str, required=True,
        help='Output similar figure file')

    opt = parser.parse_args()
    print(opt)

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]
    
    match_dir = Path(opt.match_dir)
    print('Looking for match .npz in directory \"{}\"'.format(match_dir))

    match_ratio_threshold = opt.match_ratio_threshold
    output_file = open(opt.similar_figure_file, 'w')

    for i, pair in enumerate(pairs):
        name0, name1 = pair[2:4] # index 2 and 3 are the image file name
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = match_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        
        # Load matching results
        npz = np.load(matches_path)
        num_matches = np.sum(npz['matches']>-1)
        kpts0 = npz['keypoints0'].shape[0]
        kpts1 = npz['keypoints1'].shape[0]
        min_num_keypoint = min(kpts0, kpts1)
        match_ratio = float(num_matches) / float(min_num_keypoint)
        if match_ratio < match_ratio_threshold: 
            continue
        
        
        output_file.write('{}\t{:.2f}\t{}\t{}\n'.format('\t'.join(pair), match_ratio, kpts0, kpts1))

    output_file.close()
    
