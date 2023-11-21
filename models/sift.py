import cv2
import numpy as np
import argparse
import os
import torch

def sift_feature_matching_BFMatcher(img1, img2):
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    unique_kp1 = set()
    unique_kp2 = set()

    for match in good_matches:
        unique_kp1.add(match[0].queryIdx)
        unique_kp2.add(match[0].trainIdx)

    matched_image = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good_matches, None, flags=2)
    
    desc_text = 'Brute-force matching with SIFT'
    cv2.putText(matched_image, desc_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    kpts_text= f'Keypoints: {len(unique_kp1)}:{len(unique_kp2)}'
    cv2.putText(matched_image, kpts_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    match_text = f'Matches: {len(good_matches)}'
    cv2.putText(matched_image, match_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return matched_image, len(good_matches)

def sift_feature_matching_FLANN(img1, img2):
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    unique_kp1 = set()
    unique_kp2 = set()

    for match in good_matches:
        unique_kp1.add(match[0].queryIdx)
        unique_kp2.add(match[0].trainIdx)

    matched_image = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good_matches, None, flags=2)

    desc_text = 'FLANN-based matching with SIFT'
    cv2.putText(matched_image, desc_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    kpts_text = f'Keypoints: {len(unique_kp1)}:{len(unique_kp2)}'
    cv2.putText(matched_image, kpts_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    match_text = f'Matches: {len(good_matches)}'
    cv2.putText(matched_image, match_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return matched_image, len(good_matches)

def process_video(video_path, output_dir=None, matcher=0):
    cap = cv2.VideoCapture(video_path)

    prev_frame = None
    current_frame = None
    ret, next_frame = cap.read()

    while ret:
        prev_frame = current_frame
        current_frame = next_frame
        ret, next_frame = cap.read()

        if prev_frame is None:
            continue

        if matcher == 1:
            if prev_frame is not None:
                result_prev, num_matches_prev = sift_feature_matching_FLANN(current_frame, prev_frame)
                cv2.imshow('Matches with Previous Frame', result_prev)

            if next_frame is not None:
                result_next, num_matches_next = sift_feature_matching_FLANN(current_frame, next_frame)
                cv2.imshow('Matches with Next Frame', result_next)
        else:
            if prev_frame is not None:
                result_prev, num_matches_prev = sift_feature_matching_BFMatcher(current_frame, prev_frame)
                cv2.imshow('Matches with Previous Frame', result_prev)

            if next_frame is not None:
                result_next, num_matches_next = sift_feature_matching_BFMatcher(current_frame, next_frame)
                cv2.imshow('Matches with Next Frame', result_next)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_directory(directory, output_dir, matcher=0):
    images = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg', '.mp4'))]
    for i in range(0, len(images), 2):
        if i+1 < len(images):
            img1 = cv2.imread(os.path.join(directory, images[i]))
            img2 = cv2.imread(os.path.join(directory, images[i+1]))
            
            if matcher == 1:
                result, num_matches = sift_feature_matching_FLANN(img1, img2)
            else: 
                result, num_matches = sift_feature_matching_BFMatcher(img1, img2)

            print(f'Number of matches: {num_matches}')

            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f'matched_{i}.png'), result)
            cv2.imshow('Matches', result)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


