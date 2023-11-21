#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# import sys
# print(sys.path)

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import open3d as o3d

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
from models.slam import Display, PointMap, SLAMProcessor
from models.sift import process_directory, process_video

pmap = PointMap()
display = Display()
slam_processor = SLAMProcessor(fx=3000, fy=3000, cx=640, cy=480)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--use_sift', action='store_true',
        help='Use SIFT features instead of SuperPoint')
    parser.add_argument(
        '--video', action='store_true',
        help='Test SIFT with video')
    parser.add_argument(
        '--flann', action='store_true',
        help='Test SIFT with FLANN-based matching')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    if opt.use_sift:
        if opt.flann:
            if opt.video:
                process_video('assets/driving_video/driving.mp4', matcher=1)
            else:
                process_directory(opt.input, opt.output_dir, 1)
        else:
            if opt.video:
                process_video('assets/driving_video/driving.mp4', matcher=0)
            else:
                process_directory(opt.input, opt.output_dir, 0)
    else:
     matching = Matching(config).eval().to(device)
     keys = ['keypoints', 'scores', 'descriptors']
     vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                        opt.image_glob, opt.max_length)
     frame, ret = vs.next_frame()
     assert ret, 'Error when reading the first frame (try different --input?)'

     frame_tensor = frame2tensor(frame, device)
     last_data = matching.superpoint({'image': frame_tensor})
     last_data = {k+'0': last_data[k] for k in keys}
     last_data['image0'] = frame_tensor
     last_frame = frame
     last_image_id = 0

     if opt.output_dir is not None:
         print('==> Will write outputs to {}'.format(opt.output_dir))
         Path(opt.output_dir).mkdir(exist_ok=True)

     # Create a window to display the demo.
     if not opt.no_display:
         cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
         cv2.resizeWindow('SuperGlue matches', 640*2, 480)
     else:
         print('Skipping visualization, will not show a GUI.')

     # Print the keyboard help menu.
     print('==> Keyboard control:\n'
           '\tn: select the current frame as the anchor\n'
           '\te/r: increase/decrease the keypoint confidence threshold\n'
           '\td/f: increase/decrease the match filtering threshold\n'
           '\tk: toggle the visualization of keypoints\n'
           '\tq: quit')
 
     timer = AverageTimer()

     point_cloud = o3d.geometry.PointCloud()
     print("init point_cloud", point_cloud)
     visualizer = o3d.visualization.Visualizer()
     visualizer.create_window(window_name = "3D distribution", width=700, height=600) 
     visualizer.add_geometry(point_cloud)
     while True:
         # print("Iter", iter)
         frame, ret = vs.next_frame()
         print(f"Current frame index: {vs.i}")
         # print('frame: ', frame.shape)
         # print('ret: ', ret)
         if not ret:
             print('Finished demo_superglue.py')
             break
         timer.update('data')
         stem0, stem1 = last_image_id, vs.i - 1

         frame_tensor = frame2tensor(frame, device)
         pred = matching({**last_data, 'image1': frame_tensor})
         kpts0 = last_data['keypoints0'][0].cpu().numpy()
         kpts1 = pred['keypoints1'][0].cpu().numpy()
         matches = pred['matches0'][0].cpu().numpy()
         confidence = pred['matching_scores0'][0].cpu().numpy()
         timer.update('forward')

         valid = matches > -1
         mkpts0 = kpts0[valid]
         mkpts1 = kpts1[matches[valid]]
         color = cm.jet(confidence[valid])
         text = [
             'SuperGlue',
             'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
             'Matches: {}'.format(len(mkpts0))
         ]
         k_thresh = matching.superpoint.config['keypoint_threshold']
         m_thresh = matching.superglue.config['match_threshold']
         small_text = [
             'Keypoint Threshold: {:.4f}'.format(k_thresh),
             'Match Threshold: {:.2f}'.format(m_thresh),
             'Image Pair: {:06}:{:06}'.format(stem0, stem1),
         ]
         out = make_matching_plot_fast(
             last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
             path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
         
         print(f"mkpts0.shape: {mkpts0.shape}")
         print(f"mkpts1.shape: {mkpts1.shape}")
         print('last_frame shape and type', last_frame.shape, type(last_frame))
         # print('last_frame', last_frame)
         print('frame shape', frame.shape)
         # print('frame', frame)
         print('diff between frames', np.subtract(frame, last_frame).shape)
         print('out shape', out.shape)
         # print('out', out)

         # print('OUTPUT', out)
         tri_points, pc, pose = slam_processor.process_video(last_frame, frame, mkpts0, mkpts1)
         print("tri_points shape:", tri_points.shape)
         # print("pc:", pc.shape)
         print("pose:", pose.shape)
         point_map = pmap.get_points(tri_points)
         # # print('Poiny_map:', len(point_map))
         # display.display_points3d(pc, point_cloud, visualizer)
         display.accumulate_points(point_map, pose, point_cloud, visualizer)
         last_frame = frame

         if not opt.no_display:
             cv2.imshow('SuperGlue matches', out)
             key = chr(cv2.waitKey(1) & 0xFF)
             if key == 'q':
                 vs.cleanup()
                 print('Exiting (via q) demo_superglue.py')
                 break
             elif key == 'n':  # set the current frame as anchor
                 last_data = {k+'0': pred[k+'1'] for k in keys}
                 last_data['image0'] = frame_tensor
                 last_frame = frame
                 print(f"last_frame updated at index: {vs.i - 1}")
                 last_image_id = (vs.i - 1)
             elif key in ['e', 'r']:
                 # Increase/decrease keypoint threshold by 10% each keypress.
                 d = 0.1 * (-1 if key == 'e' else 1)
                 matching.superpoint.config['keypoint_threshold'] = min(max(
                     0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                 print('\nChanged the keypoint threshold to {:.4f}'.format(
                     matching.superpoint.config['keypoint_threshold']))
             elif key in ['d', 'f']:
                 # Increase/decrease match threshold by 0.05 each keypress.
                 d = 0.05 * (-1 if key == 'd' else 1)
                 matching.superglue.config['match_threshold'] = min(max(
                     0.05, matching.superglue.config['match_threshold']+d), .95)
                 print('\nChanged the match threshold to {:.2f}'.format(
                     matching.superglue.config['match_threshold']))
             elif key == 'k':
                 opt.show_keypoints = not opt.show_keypoints

         timer.update('viz')
         timer.print()

         if opt.output_dir is not None:
             #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
             stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
             out_file = str(Path(opt.output_dir, stem + '.png'))
             print('\nWriting image to {}'.format(out_file))
             cv2.imwrite(out_file, out)

     # display.create_final_viz(visualizer)
     # visualizer.destroy_window()
     cv2.destroyAllWindows()
     vs.cleanup()
