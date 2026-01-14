"""
Script to analyze AM/PM detection issues by comparing video parsing results 
with ground truth data from the Google Spreadsheet.

This script:
1. Fetches ground truth dates from the 'Codes' sheet (column L)
2. Parses videos in videos/temp and videos/done 
3. Compares parsed AM/PM with ground truth AM/PM
4. Identifies and analyzes mismatches

Usage:
    conda activate s3
    python analyze_ampm.py
"""

import cv2
import numpy as np
import pygsheets
import vid2data as v
import os
from datetime import datetime

# Rows to exclude (1-indexed as in spreadsheet)
EXCLUDE_ROWS = set(range(14731, 14744)) | set(range(16930, 16948))


def fetch_ground_truth_dates():
    """Fetch all dates from column L of the Codes sheet."""
    print("Connecting to Google Sheets...")
    gc = pygsheets.authorize(service_file='scode.json')
    sh = gc.open("Leo's codes v3.1")
    wks = sh.worksheet_by_title('Codes')
    
    # Get column L (dates) - returns list of values
    print("Fetching dates from column L...")
    dates = wks.get_col(12)  # Column L is 12
    
    # Create dictionary mapping date -> row numbers (1-indexed)
    date_to_rows = {}
    for i, date in enumerate(dates):
        row_num = i + 1  # 1-indexed
        if row_num in EXCLUDE_ROWS:
            continue
        if date and date.strip():
            if date not in date_to_rows:
                date_to_rows[date] = []
            date_to_rows[date].append(row_num)
    
    print(f"Fetched {len(date_to_rows)} unique dates ({len(dates)} total rows)")
    return date_to_rows, dates


def extract_ampm_info(frame):
    """
    Extract the AM/PM letter from a frame and return analysis info.
    Returns dict with pixel sum and detected AM/PM, or None on error.
    """
    frame = v.resize_frame(frame)
    if frame is None:
        return None
    
    frame = frame[:, v.REMOVE:]
    date = frame[v.DATETOP:v.DATEBOTTOM, v.DATELEFT:v.DATERIGHT]
    
    h, w, _ = date.shape
    col_sums = np.sum(date, axis=2)
    black_counts = np.sum(col_sums < 100, axis=0)
    
    stop_idx = w
    stop = 0
    for x in range(w - 1, -1, -1):
        if black_counts[x] >= 15:
            stop += 1
        else:
            stop = 0
        if stop > 7:
            stop_idx = x + 8
            break
    date = date[:, :stop_idx]
    
    gray = cv2.cvtColor(date, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    indices = np.argsort(centroids[1:, 0])
    
    if len(indices) < 6:
        return None
    
    indices = np.delete(indices, [-5, -6, -1])
    stats = stats[1:]
    letters = []
    for i in range(len(indices)):
        ind = indices[i]
        if i == len(indices) - 2:
            stats[ind][2] += 1
        letters.append(v.isolate_letter(date, stats[ind], [18, 12]))
    
    if len(letters) == 0:
        return None
    
    last_letter = letters[-1]
    right_half_sum = np.sum(last_letter[:, 6:])
    full_sum = np.sum(last_letter)
    detected = 'PM' if right_half_sum < 24300 else 'AM'
    
    return {
        'right_half_sum': right_half_sum,
        'full_sum': full_sum,
        'detected': detected,
        'letter_image': last_letter
    }


def analyze_video(video_path, ground_truth_dates, max_frames=500):
    """
    Analyze a video and compare parsed dates with ground truth.
    Returns list of mismatches and summary statistics.
    """
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f"  Could not open video: {video_path}")
        return [], {}
    
    results = []
    last_date = None
    am_sums = []
    pm_sums = []
    
    for frame_num in range(max_frames):
        ret, frame = cam.read()
        if not ret:
            break
        
        frame = v.resize_frame(frame)
        if frame is None:
            print(f"  Unsupported resolution in {video_path}")
            break
        
        try:
            parsed_date = v.get_date(frame)
        except Exception as e:
            continue
        
        # Only process new scenarios
        if parsed_date == last_date or '-' in parsed_date:
            continue
        last_date = parsed_date
        
        # Get AM/PM analysis
        info = extract_ampm_info(frame)
        if info is None:
            continue
        
        detected_ampm = info['detected']
        parsed_ampm = 'PM' if 'PM' in parsed_date else 'AM'
        
        # Look up ground truth
        # The parsed_date format is like "1/4/2026 8:24AM"
        # Ground truth should match exactly
        ground_truth_ampm = None
        matching_gt_date = None
        
        # Search for this date in ground truth
        for gt_date in ground_truth_dates:
            if gt_date == parsed_date:
                ground_truth_ampm = 'PM' if 'PM' in gt_date else 'AM'
                matching_gt_date = gt_date
                break
        
        # Also check if date without AM/PM matches
        base_date = parsed_date.replace('AM', '').replace('PM', '')
        for gt_date in ground_truth_dates:
            gt_base = gt_date.replace('AM', '').replace('PM', '')
            if gt_base == base_date:
                ground_truth_ampm = 'PM' if 'PM' in gt_date else 'AM'
                matching_gt_date = gt_date
                break
        
        # Track pixel sums by ground truth
        if ground_truth_ampm == 'AM':
            am_sums.append(info['right_half_sum'])
        elif ground_truth_ampm == 'PM':
            pm_sums.append(info['right_half_sum'])
        
        # Check for mismatch
        if ground_truth_ampm and ground_truth_ampm != detected_ampm:
            results.append({
                'frame': frame_num,
                'parsed_date': parsed_date,
                'ground_truth_date': matching_gt_date,
                'ground_truth_ampm': ground_truth_ampm,
                'detected_ampm': detected_ampm,
                'pixel_sum': info['right_half_sum'],
                'letter_image': info['letter_image']
            })
    
    cam.release()
    
    stats = {
        'am_sums': am_sums,
        'pm_sums': pm_sums
    }
    
    return results, stats


def main():
    """Main analysis function."""
    # Fetch ground truth
    ground_truth_dates, all_dates = fetch_ground_truth_dates()
    
    # Analyze videos in temp folder
    temp_dir = 'videos/temp'
    done_dir = 'videos/done'
    
    all_mismatches = []
    all_am_sums = []
    all_pm_sums = []
    
    print("\n" + "="*60)
    print("Analyzing videos in videos/temp")
    print("="*60)
    
    if os.path.exists(temp_dir):
        for filename in sorted(os.listdir(temp_dir)):
            if filename.endswith(('.mp4', '.mov')):
                print(f"\nProcessing: {filename}")
                video_path = os.path.join(temp_dir, filename)
                mismatches, stats = analyze_video(video_path, ground_truth_dates)
                all_mismatches.extend(mismatches)
                all_am_sums.extend(stats.get('am_sums', []))
                all_pm_sums.extend(stats.get('pm_sums', []))
                
                if mismatches:
                    print(f"  Found {len(mismatches)} AM/PM mismatches:")
                    for m in mismatches:
                        print(f"    Frame {m['frame']}: parsed={m['parsed_date']}, "
                              f"GT={m['ground_truth_date']}, "
                              f"sum={m['pixel_sum']:.0f}")
                else:
                    print(f"  No AM/PM mismatches found")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nTotal AM/PM mismatches: {len(all_mismatches)}")
    
    if all_am_sums:
        print(f"\nAM pixel sums (ground truth AM):")
        print(f"  Min: {min(all_am_sums):.0f}")
        print(f"  Max: {max(all_am_sums):.0f}")
        print(f"  Mean: {np.mean(all_am_sums):.0f}")
        print(f"  Count: {len(all_am_sums)}")
    
    if all_pm_sums:
        print(f"\nPM pixel sums (ground truth PM):")
        print(f"  Min: {min(all_pm_sums):.0f}")
        print(f"  Max: {max(all_pm_sums):.0f}")
        print(f"  Mean: {np.mean(all_pm_sums):.0f}")
        print(f"  Count: {len(all_pm_sums)}")
    
    print(f"\nCurrent threshold: 24300")
    print("(Below threshold = PM, Above threshold = AM)")
    
    # Save mismatch letter images for analysis
    if all_mismatches:
        os.makedirs('out/ampm_mismatches', exist_ok=True)
        for i, m in enumerate(all_mismatches):
            filename = f"out/ampm_mismatches/{i:03d}_GT_{m['ground_truth_ampm']}_DET_{m['detected_ampm']}_sum{m['pixel_sum']:.0f}.png"
            cv2.imwrite(filename, m['letter_image'])
        print(f"\nSaved {len(all_mismatches)} mismatch letter images to out/ampm_mismatches/")


if __name__ == '__main__':
    main()
