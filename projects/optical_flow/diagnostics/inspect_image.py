#!/usr/bin/env python3
"""Consolidated image dimensions, abrupt column edges and repeated-column diagnostics."""
import argparse
import json
from PIL import Image


def inspect(path):
    with Image.open(path) as image:
        image.load()
        rgb = image.convert('RGB')
        w, h = rgb.size
        pixels = rgb.load()
        gray = rgb.convert('L').load()
        edges = [(x, sum(abs(pixels[x,y][c] - pixels[x-1,y][c]) for y in range(h) for c in range(3)) / (h*3)) for x in range(1,w)]
        offsets = [(d, sum(abs(gray[x,y] - gray[x+d,y]) for y in range(h) for x in range(w-d)) / (h*(w-d))) for d in range(1, min(w, w//2+50))]
        return {'size': [w,h], 'mode': image.mode, 'strongest_column_edges': sorted(edges,key=lambda x:x[1],reverse=True)[:5],
                'most_similar_column_offsets': sorted(offsets,key=lambda x:x[1])[:5],
                'note': 'Image-only diagnostic; these statistics do not establish optical-flow correctness.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    args=parser.parse_args()
    print(json.dumps(inspect(args.image)))
