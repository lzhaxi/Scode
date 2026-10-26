import cv2
import sys
import numpy as np
import vid2data

def test_bb():
    info = cv2.imread('debug_haz_60.png')
    gray = cv2.cvtColor(info, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    valid_stats = [s for s in stats if s[3] >= 10]
    
    # We also need to do the exact steps vid2data does
    # remove the percent symbol, which is at the max x value
    length = len(stats)
    i = 0
    while i < length:
        if stats[i][3] < 10:
            stats = np.delete(stats, i, axis=0)
            centroids = np.delete(centroids, i, axis=0)
            length -= 1
        else:
            i += 1

    max_idx = np.argmax(centroids[:, 0])
    stats = np.delete(stats, max_idx, axis=0)
    centroids = np.delete(centroids, max_idx, axis=0)
    
    digits = []
    for i in range(1, len(stats)):
        digits.append(vid2data.isolate_letter(info, stats[i], [24, 21]))
    digits = [x for _, x in sorted(zip(centroids[1:, 0], digits))]
    
    # The last digit should be the slash
    slash_img = digits[-1]
    cv2.imwrite('ground_truths/hazard_data/empty.png', slash_img)
    print("Saved ground_truths/hazard_data/empty.png")
    
if __name__ == '__main__':
    test_bb()
