from PIL import Image
import sys
img = Image.open('logs/flow_frames/latest/frame_001.png')
w, h = img.size
pix = img.load()
res = []
for offset in range(10, w//2 + 50):
    diff_sum = 0
    for y in range(h):
        for x in range(w - offset):
            r1, g1, b1 = pix[x, y]
            r2, g2, b2 = pix[x + offset, y]
            gray1 = int(r1*0.3 + g1*0.6 + b1*0.1)
            gray2 = int(r2*0.3 + g2*0.6 + b2*0.1)
            diff_sum += abs(gray1 - gray2)
    diff = diff_sum / (h * (w - offset))
    res.append((offset, diff))
res.sort(key=lambda x: x[1])
print('Top 5 most matching offsets:', res[:5])
