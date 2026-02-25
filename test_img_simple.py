import cv2
import sys
img = cv2.imread('logs/flow_frames/latest/frame_008.png')
if img is not None:
    print('Shape:', img.shape)
