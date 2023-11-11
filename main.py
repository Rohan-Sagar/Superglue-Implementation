import cv2 as cv
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import re

class FeatureMatching():
    def __init__(self, img1, img2):
        self.sift = cv.SIFT_create()
        self.img1 = img1
        self.img2 = img2

    def extract(self):
        kp1, des1 = self.sift.detectAndCompute(self.img1, None)
        kp2, des2 = self.sift.detectAndCompute(self.img2, None) 
        return kp1, kp2, des1, des2

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def process_images(input_directory, output_directory):
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=numerical_sort) 
    # print(image_files)

    for i in range(0, len(image_files) - 1, 2):
        img1_path = os.path.join(input_directory, image_files[i])
        img2_path = os.path.join(input_directory, image_files[i+1])
        img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

        fe = FeatureMatching(img1, img2)
        kp1, kp2, des1, des2 = fe.extract()
        pts1 = [p.pt for p in kp1]
        pts2 = [p.pt for p in kp2]
    
        for i in pts1:
            cv.circle(img1, (round(i[0]), round(i[1])), 4, (0, 0, 255), -1)

        for j in pts2:
            cv.circle(img2, (round(j[0]), round(j[1])), 4, (0, 255, 0), -1)
    
        # _, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(img1)
        # axarr[1].imshow(img2)
        # plt.show()
        
        print(img1.shape, img2.shape, img1_path, img2_path)
        combined_image = np.hstack((img1, img2))

        output_path = os.path.join(output_directory, f'combined_{i}.png')
        cv.imwrite(output_path, combined_image)


def process_captured_videos(directory):
    # video_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # video_files.sort() 
    pass


def main():
    parser = argparse.ArgumentParser(description="Feature Matching on Images or Video")
    parser.add_argument('--video', action='store_true', help='Process video instead of images')
    args = parser.parse_args()

    if args.video:
        process_captured_videos('assets/Hub_apt/videos')
    else:
        process_images('assets/driving', 'output/driving')
