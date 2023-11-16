import cv2 as cv
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from models.superglue import KeypointEncoder, AttentionalGNN, log_optimal_transport, compute_similarity
from models.utils import create_edges, draw_matches, draw_keypoints, numerical_sort 

input_dim = 2               # x and y coordinate
hidden_dim = 128 
output_dim = 128            # 128-dimensional vector

class FeatureExtractor():
    def __init__(self, image):
        self.sift = cv.SIFT_create()
        self.image = image

    def extract(self, max_kpts):
        kpts, des = self.sift.detectAndCompute(self.image, None)
        print("Kpts:", kpts)
        if len(kpts) > max_kpts:
            sort_kpts = sorted(zip(kpts, des), key = lambda x: x[0].response, reverse=True)
            sort_kpts = sort_kpts[:max_kpts]
            kpts, des = zip(*sort_kpts)
            kpts, des = list(kpts), np.array(des)

        return kpts, des

def process_single_image(image_path, max_kpts = 100):
    img = cv.imread(image_path) #, cv.IMREAD_GRAYSCALE)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (881, 400))
    print("Image shape:", img.shape)
    fe = FeatureExtractor(img)
    kpts, des = fe.extract(max_kpts = max_kpts)
    print("Num kpts:", len(kpts))
    ke = KeypointEncoder(kpts, des)
    output = ke()
    img_kpts = draw_keypoints(img, kpts)
    return img_kpts, kpts, des, output 

def cross_check(matches_forward, matches_backward):
    print("Forward Matches:", matches_forward)
    print("\n")
    print("Backward Matches:", matches_backward)
    mutual_matches = torch.where(matches_forward.gather(1, matches_backward) == torch.arange(matches_forward.size(0)).unsqueeze(1).to(matches_forward.device),
                                 matches_backward, torch.full_like(matches_backward, -1))
    print("Cross-checked Matches:", mutual_matches)
    return mutual_matches

def lowe_ratio_test(scores, ratio_threshold=0.75):
    top_two = scores.topk(2, dim=2, largest=True, sorted=True)[0]
    print("Top_two scores:", top_two)
    mask = top_two[:, :, 0] < ratio_threshold * top_two[:, :, 1]
    mask = mask.unsqueeze(2).expand_as(scores)
    scores_lowe = torch.where(mask, scores, torch.full_like(scores, -1e5))
    print("Scores after Lowe's ratio test:", scores_lowe)
    return scores_lowe

def process_images(input_directory, output_directory):
    print("Printing new combination!!!!", "\n")
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))], key=numerical_sort)
    gnn = AttentionalGNN(num_features=128, layer_count=3)

    for i in range(0, len(image_files) - 1, 2):
        img1_path = os.path.join(input_directory, image_files[i])
        img2_path = os.path.join(input_directory, image_files[i+1])
        img1, kp1, desc1, ke1 = process_single_image(img1_path)
        img2, kp2, desc2, ke2 = process_single_image(img2_path)

        desc1_tensor = torch.from_numpy(desc1).float()
        desc2_tensor = torch.from_numpy(desc2).float()

        num_kpts1 = len(kp1)
        num_kpts2 = len(kp2)

        edges = create_edges(kp1, kp2)

        output = gnn(desc1_tensor, desc2_tensor, edges)
        print("GNN output:", output, output.shape)
        scores = compute_similarity(output)
        print("Compute_similarity_scores:", scores)
        print("Scores shape:", scores.shape)

        alpha = torch.nn.Parameter(torch.tensor(1.))
        iters = 20
        optimal_matching = log_optimal_transport(scores, alpha, iters)[:, :-1, :-1]
        print("Optimal transport matching:", optimal_matching)

        matches_forward = optimal_matching.argmax(dim=2)

        backward_scores = compute_similarity(output).transpose(1, 2)
        matches_backward = backward_scores.argmax(dim=2)

        matches_cross_checked = cross_check(matches_forward, matches_backward)
        print("Matches after cross-checking:", matches_cross_checked)

        scores_lowe = lowe_ratio_test(scores)
        matches_lowe = scores_lowe.argmax(dim=2)
        print("Matches_lowe:", matches_lowe)

        matches_combined = torch.where(matches_cross_checked == matches_lowe, matches_cross_checked, torch.full_like(matches_cross_checked, -1))
        print("Combined Matches after cross-check and Lowe's test:", matches_combined)

        valid_matches_mask = (matches_combined >= 0)
        print("Number of valid combined matches:", valid_matches_mask.sum().item())

        combined_image_with_matches = draw_matches(img1, kp1, img2, kp2, matches_combined, -1)
        output_path = os.path.join(output_directory, f'combined_{i}.png')
        cv.imwrite(output_path, combined_image_with_matches)


def process_video(video_path, output_directory):
    cap = cv.VideoCapture(video_path)
    extractor = FeatureExtractor(None)  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # vid = cv.imread(frame)
        extractor.image = frame
        keypoints, _ = extractor.extract()
        frame_with_keypoints = draw_keypoints(frame, keypoints)
        cv.imshow('frame', frame_with_keypoints)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
