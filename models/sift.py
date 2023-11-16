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
        if len(kpts) > max_kpts:
            sort_kpts = sorted(zip(kpts, des), key = lambda x: x[0].response, reverse=True)
            sort_kpts = sort_kpts[:max_kpts]
            kpts, des = zip(*sort_kpts)
            kpts, des = list(kpts), np.array(des)

        return kpts, des

def process_single_image(image_path, max_kpts = 100):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (881, 400))
    print("Image shape:", img.shape)
    fe = FeatureExtractor(img)
    kpts, des = fe.extract(max_kpts = max_kpts)
    print("Num kpts:", len(kpts))
    ke = KeypointEncoder(kpts, des)
    output = ke()
    img_kpts = draw_keypoints(img, kpts)
    return img_kpts, kpts, des, output 

def process_images(input_directory, output_directory):
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))], key=numerical_sort)
    # print(len(image_files)) 
    gnn = AttentionalGNN(num_features=128, layer_count=3)

    for i in range(0, len(image_files) - 1, 2):
        # nodes = []
        img1_path = os.path.join(input_directory, image_files[i])
        img2_path = os.path.join(input_directory, image_files[i+1])
        print("\n")
        print(img1_path, img2_path)
        img1, kp1, desc1, ke1 = process_single_image(img1_path)
        img2, kp2, desc2, ke2 = process_single_image(img2_path)
        print("Encoded keypoints shape:", ke1.shape, ke2.shape)   # torch.Size([num_kpts, num_features])

        # print("Desc1:", desc1)
        # print(type(desc1), type(desc2))

        desc1_tensor = torch.from_numpy(desc1).float()
        desc2_tensor = torch.from_numpy(desc2).float()

        # print(type(kp1), type(kp2))
        num_kpts1 = len(kp1)
        num_kpts2 = len(kp2)
        print("Num kpts1:", num_kpts1, "Num kpts2:", num_kpts2)

        edges = create_edges(kp1, kp2)
        print("Edge type:", type(edges),
              "Edge len:", edges.shape)
              # "Edges:", edges)

        output = gnn(desc1_tensor, desc2_tensor, edges)
        print('GNN output:', output)
        scores = compute_similarity(output)
        print("Scores shape", scores.shape)
        print("Scores", scores)

        b, m, n = scores.shape  # b: batch size, m: number of keypoints in image 1, n: number of keypoints in image 2

        alpha = torch.nn.Parameter(torch.tensor(1.))
        iters = 20 
        optimal_matching = log_optimal_transport(scores, alpha, iters)[:, :-1, :-1]
        print("Optimal matching:", optimal_matching)

        # Get the indices of the best matches for each keypoint in the first image
        matches_indices = optimal_matching.argmax(dim=2)
        # Get the match scores for these indices
        match_scores = torch.gather(optimal_matching, 2, matches_indices.unsqueeze(2)).squeeze(2)

        no_match_bin_index = -1
        probability_threshold = 0.000005
        match_threshold = torch.log(torch.tensor(probability_threshold))
        print("Match threshold:", match_threshold)
        matches = torch.where(match_scores > match_threshold, matches_indices, torch.tensor(no_match_bin_index))
        print("Matches", matches)
        print("Number of matches:", (matches != no_match_bin_index).sum().item())

        combined_image_with_matches = draw_matches(img1, kp1, img2, kp2, matches, no_match_bin_index)
        output_path = os.path.join(output_directory, f'combined_{i}.png')
        cv.imwrite(output_path, combined_image_with_matches)

        # combined_image = np.hstack((img1, img2))
        # output_path = os.path.join(output_directory, f'combined_{i}.png')
        # cv.imwrite(output_path, combined_image)

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
