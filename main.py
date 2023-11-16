import argparse
from models.sift import process_images, process_video

def main():
    parser = argparse.ArgumentParser(description="Feature Matching on Images or Video")
    parser.add_argument('--video', action='store_true', help='Process video instead of images')
    args = parser.parse_args()

    if args.video:
        process_video('assets/driving/video/pexels_videos_1666547 (720p).mp4', 'output/driving/video')
    else:
        process_images('assets/Hub_apt/images', 'output/driving/images')

