from sklearn.decomposition import PCA
import cv2 as cv
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from models.superglue import KeypointsMLP

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

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

mlp = KeypointsMLP(input_dim, hidden_dim, output_dim)

def process_single_image(image_path):
    img = cv.imread(image_path) # , cv.IMREAD_GRAYSCALE)
    # print(img.shape)
    fe = FeatureExtractor(img)
    kpts, des = fe.extract()
    # print("desc", des)
    
    # converting kpts to tensor
    kpts_pos = np.array([kp.pt for kp in kpts], dtype=np.float32)
    kpts_pos_tensor = torch.tensor(kpts_pos)
    kpt_embeddings = mlp(kpts_pos_tensor)
    print(type(des), type(kpt_embeddings))

    # embeddings_np = kpt_embeddings.detach().numpy()
    #
    # pca = PCA(n_components=2)
    # reduced_embeddings = pca.fit_transform(embeddings_np)
    #
    # plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.title('PCA of Keypoint Embeddings')
    # plt.show()

    img_kpts = draw_keypoints(img, kpts)
    return img_kpts, kpts, des, kpt_embeddings

def process_images(input_directory, output_directory):
    image_files = sorted([f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))], key = numerical_sort)

    for i in range(0, len(image_files) - 1, 2):
        img1_path = os.path.join(input_directory, image_files[i])
        img2_path = os.path.join(input_directory, image_files[i+1])

        img1, kp1, des1, embed1 = process_single_image(img1_path)
        img2, kp2, des2, embed1 = process_single_image(img2_path)

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
