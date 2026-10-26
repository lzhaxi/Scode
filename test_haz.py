import cv2
import sys
import os
import numpy as np
import vid2data

def test_video(filename):
    print(f"\n--- Testing {filename} ---")
    cam = cv2.VideoCapture('videos/' + filename)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    unique_hazards = set()
    
    for fnum in range(0, total_frames, 30):
        cam.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = cam.read()
        if not ret: continue
        
        needs_resize = vid2data.SUPPORTED_RESOLUTIONS.get(frame.shape) is not None
        if needs_resize:
            frame = vid2data.resize_frame(frame)
            
        if frame is None:
            continue
        
        info = frame[:, vid2data.REMOVE:]
        haz = info[vid2data.HAZTOP:vid2data.HAZBOTTOM, vid2data.HAZLEFT+vid2data.HAZLANG['en']:vid2data.HAZRIGHT]
        
        gray = cv2.cvtColor(haz, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(binary) < 50:
            continue
            
        _, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
        valid_stats = [s for s in stats if s[3] >= 10]
        
        try:
            hazard_str = vid2data.get_hazard(info)
            if len(hazard_str) > 0 and hazard_str not in unique_hazards:
                print(f"Frame {fnum:4d} | Valid stats: {len(valid_stats)} | Found Hazard: '{hazard_str}'")
                unique_hazards.add(hazard_str)
        except Exception:
            pass

if __name__ == '__main__':
    for f in sorted(os.listdir('videos')):
        if f.endswith('.mp4'):
            test_video(f)
