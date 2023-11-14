import cv2 as cv
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from models.superglue import KeypointEncoder, AttentionalGNN
from models.utils import pad_tensor, create_edges 

input_dim = 2    # x and y coordinate
hidden_dim = 128 
output_dim = 128 # 128-dimensional vector

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

def draw_keypoints(img, kpts):
    for coord in kpts:
        cv.circle(img, (round(coord.pt[0]), round(coord.pt[1])), 2, (0, 255, 0), -1)
    return img 

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def process_single_image(image_path, max_kpts = 100):
    img = cv.imread(image_path) # , cv.IMREAD_GRAYSCALE)
    # print(img.shape)
    fe = FeatureExtractor(img)
    kpts, des = fe.extract(max_kpts = max_kpts)
    # print(len(kpts))
    ke = KeypointEncoder(kpts, des)
    output = ke(kpts, des)
    img_kpts = draw_keypoints(img, kpts)
    return img_kpts, kpts, des, output 

def process_images(input_directory, output_directory):
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))], key = numerical_sort)
    # print(len(image_files)) 
    gnn = AttentionalGNN(num_features = 128)

    for i in range(0, len(image_files) - 1, 2):
        nodes = []
        img1_path = os.path.join(input_directory, image_files[i])
        img2_path = os.path.join(input_directory, image_files[i+1])
        # print(img1_path, img2_path)
        img1, kp1, des1, ke1 = process_single_image(img1_path)
        img2, kp2, des2, ke2 = process_single_image(img2_path)
        print("Encoded keypoints shape:", ke1.shape, ke2.shape)
        max_kpts = max(ke1.size(0), ke2.size(0))

        ke1_new = pad_tensor(ke1, max_kpts)
        ke2_new = pad_tensor(ke2, max_kpts)
        nodes = torch.cat([ke1_new, ke2_new], dim=0)
        # print(type(kp1), type(kp2))
        # num_kpts1 = len(kp1)
        # num_kpts2 = len(kp2)
        edges = create_edges(kp1, kp2)
        # print(type(edges), len(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # print(edge_index.shape)
        print("Nodes shape:", nodes.shape)
        print("Edge index shape:", edge_index.shape)
        print("Edge index content:", edge_index)

        output = gnn(nodes, edge_index)

        combined_image = np.hstack((img1, img2))
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
