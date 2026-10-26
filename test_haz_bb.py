import cv2
import sys
import numpy as np
import vid2data

def test_bb():
    info = cv2.imread('debug_haz_0.png')
    gray = cv2.cvtColor(info, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    valid_stats = [s for s in stats if s[3] >= 10]
    
    print(f"Total valid stats: {len(valid_stats)}")
    for i, s in enumerate(valid_stats):
        print(f"Valid Stat {i}: x={s[0]:3d}, y={s[1]:3d}, w={s[2]:3d}, h={s[3]:3d}, area={s[4]:3d}")

if __name__ == '__main__':
    test_bb()
