from scipy.spatial import KDTree
import numpy as np
import torch
import re
import cv2 as cv

def create_edges(kpts1, kpts2):
    self_edges = []
    cross_edges = []

    num_kpts1 = len(kpts1)
    num_kpts2 = len(kpts2)

    for i in range(num_kpts1):
        for j in range(i + 1, num_kpts1):
            self_edges.append((i, j))

    for i in range(num_kpts2):
        for j in range(i + 1, num_kpts2):
            self_edges.append((num_kpts1 + i, num_kpts1 + j))

    for i in range(num_kpts1):
        for j in range(num_kpts2):
            cross_edges.append((i, num_kpts1 + j))

    print("Num self edges", len(self_edges))
    print("Num cross edges", len(cross_edges))

    all_edges = self_edges + cross_edges
    edge_tensor = torch.tensor(all_edges, dtype=torch.long)

    return edge_tensor


def draw_keypoints(img, kpts):
    for coord in kpts:
        cv.circle(img, (round(coord.pt[0]), round(coord.pt[1])), 2, (0, 255, 0), -1)
    return img 


def draw_matches(img1, kp1, img2, kp2, matches, no_match_bin_index):
    img1_with_kp = draw_keypoints(img1.copy(), kp1)
    img2_with_kp = draw_keypoints(img2.copy(), kp2)

    combined_image = np.hstack((img1_with_kp, img2_with_kp))

    # Offset for keypoints in the second image
    offset = img1.shape[1]

    # Find the best match for each keypoint
    best_matches = matches.argmax(dim=1)

    for i, match_index in enumerate(best_matches):
        if 0 <= match_index < len(kp2):  # Ensure the match_index is valid
            pt1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
            pt2 = (int(kp2[match_index].pt[0]) + offset, int(kp2[match_index].pt[1]))
            cv.line(combined_image, pt1, pt2, (0, 0, 255), 1)

    return combined_image

# def draw_matches(img1, kp1, img2, kp2, matches, no_match_bin_index):
#     img1_with_kp = draw_keypoints(img1.copy(), kp1)
#     img2_with_kp = draw_keypoints(img2.copy(), kp2)
#
#     combined_image = np.hstack((img1_with_kp, img2_with_kp))
#     
#     print('Matches here:', matches)
#
#     for i, j in enumerate(matches):
#         # print("Printing j:", j)
#         # if j.item() != no_match_bin_index:
#             pt1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
#             pt2 = (int(kp2[j].pt[0]) + img1.shape[1], int(kp2[j].pt[1]))
#             cv.line(combined_image, pt1, pt2, (255, 255, 204), 2)
#
#     return combined_image

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
