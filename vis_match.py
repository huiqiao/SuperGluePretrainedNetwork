from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2


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
        '--input_dir', type=str, required=True,
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--match_dir', type=str, required=True,
        help='Path to the directory in which the .npz results')
    parser.add_argument(
        '--output_vis_dir', type=str, required=True,
        help='Path to the directory in which the output visualization image results')

    opt = parser.parse_args()
    print(opt)

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]
    
    input_dir = Path(opt.input_dir)
    print('Looking for image in directory \"{}\"'.format(input_dir))
    match_dir = Path(opt.match_dir)
    print('Looking for match .npz in directory \"{}\"'.format(match_dir))
    output_vis_dir = Path(opt.output_vis_dir)
    output_vis_dir.mkdir(exist_ok=True, parents=True)

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = match_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        
        # Load matching results
        npz = np.load(matches_path)
        num_matches = np.sum(npz['matches']>-1)
        kpts0 = npz['keypoints0'].shape[0]
        kpts1 = npz['keypoints1'].shape[0]
        min_num_keypoint = min(kpts0, kpts1)
        match_ratio = float(num_matches) / float(min_num_keypoint)
        if match_ratio < 0.1: 
            continue
        viz_path = output_vis_dir / '{:.2f}_{}_{}_matches.png'.format(match_ratio, stem0, stem1)

        # Load the image pair.
        image0 = cv2.imread(str(input_dir / name0), cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(str(input_dir / name1), cv2.IMREAD_GRAYSCALE)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        
        # Visualize
        H0, W0 = image0.shape
        H1, W1 = image1.shape
        H, W = max(H0, H1), W0 + W1 + 10

        out = 255*np.ones((H, W), np.uint8)
        out[:H0, :W0] = image0
        out[:H1, W0+10:] = image1
        out = np.stack([out]*3, -1)

        # Scale factor for consistent visualization across scales.
        sc = min(H / 640., 2.0)

        # Big text.
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(kpts0, kpts1),
            'Matches: {}'.format(num_matches),
        ]
        Ht = int(30 * sc)  # text height
        txt_color_fg = (255, 255, 255)
        txt_color_bg = (0, 0, 0)
        for i, t in enumerate(text):
            cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

        cv2.imwrite(str(viz_path), out)

            

