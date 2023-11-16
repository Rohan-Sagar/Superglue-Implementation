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
            # self_edges.append((j, i))

    for i in range(num_kpts2):
        for j in range(i + 1, num_kpts2):
            self_edges.append((num_kpts1 + i, num_kpts1 + j))
            # self_edges.append((num_kpts1 + j, num_kpts1 + i))

    for i in range(num_kpts1):
        for j in range(num_kpts2):
            cross_edges.append((i, num_kpts1 + j))
            # cross_edges.append((num_kpts1 + j, i))

    print("Num self edges", len(self_edges))
    print("Num cross edges", len(cross_edges))

    all_edges = self_edges + cross_edges
    edge_tensor = torch.tensor(all_edges, dtype=torch.long)

    return edge_tensor

def draw_keypoints(img, kpts):
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    color = (0, 0, 0)
    thickness = 1

    for kp in kpts:
        cv.circle(img, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), -1)
        text_offset_x = 10
        text_offset_y = 10
        text_position = (int(kp.pt[0]) + text_offset_x, int(kp.pt[1]) + text_offset_y)
        text = f"({int(kp.pt[0])}, {int(kp.pt[1])})"
        cv.putText(img, text, text_position, font, fontScale, color, thickness, cv.LINE_AA)
    return img

def draw_matches(img1, kp1, img2, kp2, matches, no_match_bin_index):
    img1_with_kp = draw_keypoints(img1.copy(), kp1)
    img2_with_kp = draw_keypoints(img2.copy(), kp2)

    combined_image = np.hstack((img1_with_kp, img2_with_kp))
 
    offset = img1.shape[1]
   
    best_matches = matches.argmax(dim=1)

    for i, match_index in enumerate(best_matches):
        if 0 <= match_index < len(kp2):
            pt1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
            pt2 = (int(kp2[match_index].pt[0]) + offset, int(kp2[match_index].pt[1]))
            print("Point1:", pt1)
            print("Point2:", pt2)
            cv.line(combined_image, pt1, pt2, (204, 255, 255), 1)

    # combined_image = cv.drawMatches(img1, kp1, img2, kp2, best_matches, None)
    return combined_image

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
