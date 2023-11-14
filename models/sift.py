import cv2 as cv
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from models.superglue import KeypointEncoder, MultiplexGNN 

input_dim = 2    # x and y coordinate
hidden_dim = 128 
output_dim = 256 # 256-dimensional vector

class FeatureExtractor():
    def __init__(self, image):
        self.sift = cv.SIFT_create()
        self.image = image

    def extract(self):
        kpts, des = self.sift.detectAndCompute(self.image, None)
        return kpts, des

def draw_keypoints(img, kpts):
    for coord in kpts:
        cv.circle(img, (round(coord.pt[0]), round(coord.pt[1])), 2, (0, 255, 0), -1)
    return img

def create_edges(num_kpts1, num_kpts2):
    self_edges_img1 = [(i, j) for i in range(num_kpts1) for j in range(num_kpts1) if i != j]
    self_edges_img2 = [(i, j) for i in range(num_kpts2) for j in range(num_kpts2) if i != j]
    cross_edges = [(i, j) for i in range(num_kpts1) for j in range(num_kpts1, num_kpts1 + num_kpts2)]
    return self_edges_img1 + self_edges_img2 + cross_edges

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def process_single_image(image_path):
    img = cv.imread(image_path) # , cv.IMREAD_GRAYSCALE)
    # print(img.shape)
    fe = FeatureExtractor(img)
    kpts, des = fe.extract()
    print(kpts)
    ke = KeypointEncoder(kpts, des)
    output = ke(kpts, des)
    img_kpts = draw_keypoints(img, kpts)
    return img_kpts, kpts, des, output 

# Padding for same size tensors
def pad_tensor(tensor, target_size):
    padding_size = target_size - tensor.size(0)
    if padding_size > 0:
        padding = torch.zeros(padding_size, tensor.size(1))
        padded_tensor = torch.cat([tensor, padding], dim=0)
        return padded_tensor
    return tensor

def process_images(input_directory, output_directory):
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))], key = numerical_sort)

    for i in range(0, len(image_files) - 1, 2):
        nodes = []
        img1_path = os.path.join(input_directory, image_files[i])
        img2_path = os.path.join(input_directory, image_files[i+1])

        img1, kp1, des1, ke1 = process_single_image(img1_path)
        img2, kp2, des2, ke2 = process_single_image(img2_path)
        max_kpts = max(ke1.size(0), ke2.size(0))

        ke1_new = pad_tensor(ke1, max_kpts)
        ke2_new = pad_tensor(ke2, max_kpts)

        nodes = torch.stack([ke1_new, ke2_new])
        num_kpts1 = len(kp1)
        num_kpts2 = len(kp2)
        edges = create_edges(num_kpts1, num_kpts2)
        print(edges)
        # print(nodes)
        gnn = MultiplexGNN(nodes, edges)
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
