import cv2
import numpy as np

img = cv2.imread('logs/flow_frames/latest/frame_008.png')
w = img.shape[1]
h = img.shape[0]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = []
for offset in range(10, w//2 + 50):
    diff = np.mean(np.abs(gray[:, :-offset].astype(np.int32) - gray[:, offset:].astype(np.int32)))
    res.append((offset, diff))

res.sort(key=lambda x: x[1])
print('Top 5 most matching offsets:', res[:5])
