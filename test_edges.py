import cv2
import numpy as np

img = cv2.imread('logs/flow_frames/latest/frame_008.png')
w = img.shape[1]
h = img.shape[0]

blocks_x = []
for i in range(1, w):
    diff = np.mean(np.abs(img[:, i].astype(int) - img[:, i-1].astype(int)))
    if diff > 30:
        blocks_x.append((i, diff))

print(f'Sudden column changes: {blocks_x}')
