from PIL import Image
import sys
img = Image.open('logs/flow_frames/latest/frame_001.png')
w, h = img.size
pix = img.load()
for i in range(1, w):
    diff = sum(abs(pix[i, y][0] - pix[i-1, y][0]) + abs(pix[i, y][1] - pix[i-1, y][1]) + abs(pix[i, y][2] - pix[i-1, y][2]) for y in range(h)) / h
    if diff > 60:
        print('Sudden change at x=', i, 'diff=', diff)
print('Done!')
