import cv2
import numpy as np

for fname in ['frame_004.png', 'frame_006.png', 'frame_008.png']:
    try:
        img = cv2.imread(f'logs/flow_frames/latest/{fname}')
        if img is None: continue
        
        diff = np.mean(np.abs(img[:, 1:].astype(float) - img[:, :-1].astype(float)), axis=(0,2))
        top_diff = sorted([(i+1, d) for i, d in enumerate(diff)], key=lambda x: x[1], reverse=True)[:3]
        print(f'{fname} top sudden x-edges:', top_diff)
    except Exception as e:
        print(e)
