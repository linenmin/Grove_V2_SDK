from PIL import Image
import sys
img = Image.open('logs/flow_frames/latest/frame_001.png')
print(img.size)
img.save('logs/flow_frames/latest/frame_001_check.jpg')
