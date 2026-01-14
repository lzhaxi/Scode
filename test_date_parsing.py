"""
Tests to diagnose why dates with 2-digit hours aren't being read correctly.

The video 20250722-01K0TKVF7Z78ZJAC32A2S97DT3-2A1FC1DA-BD9B-4895-A912-1BFB99244A8A - /❨.mov
appears to have issues with the date parsing in early frames.

ROOT CAUSE IDENTIFIED:
======================
The get_date function uses a background removal algorithm that looks for 7+ consecutive 
columns with ≥15 black pixels to determine where to cut off the date region.

With 2-digit hours (10:XX, 11:XX, 12:XX), there's a larger gap between the year (2025) 
and the time portion. This gap creates a region of 8+ consecutive black columns, which 
triggers the premature cutoff.

Example from frame analysis:
  - Date: 7/4/2025 10:23PM  
  - Raw region width: 183 pixels
  - Cutoff happens at x=90 (loses "10:23PM")
  - Region x=83-90 has 8 consecutive columns with ≥15 black pixels

The algorithm was designed assuming single-digit hours where the gap is smaller.

FIX OPTIONS:
1. Increase the consecutive column threshold from 7 to a higher value (e.g., 12-15)
2. Start the black column search from the RIGHT side instead of left (find trailing black)
3. Look for the AM/PM pattern first to determine expected date length
4. Use a minimum expected width before allowing cutoff (e.g., don't cut before x=150)
"""

import cv2
import numpy as np
import os
import sys


# Import from vid2data
from vid2data import (
    get_date, get_date_helper, isolate_letter, ssim, get_cached_image,
    REMOVE, DATELEFT, DATERIGHT, DATETOP, DATEBOTTOM
)


def extract_date_region(frame, lang='en'):
    """Extract and process the date region from a frame for debugging."""
    frame = frame[:, REMOVE:]
    date = frame[DATETOP:DATEBOTTOM, DATELEFT:DATERIGHT]
    
    h, w, _ = date.shape
    
    # Vectorized pixel counting
    col_sums = np.sum(date, axis=2)
    black_counts = np.sum(col_sums < 100, axis=0)
    
    # Find cutoff point
    stop_idx = w
    stop = 0
    for x in range(w):
        if black_counts[x] >= 15:
            stop += 1
        else:
            stop = 0
        if stop > 7:
            stop_idx = x
            break
    
    date = date[:, :stop_idx]
    return date


def extract_date_region_fixed(frame, lang='en'):
    """
    FIXED version: Extract and process the date region from a frame.
    
    This version searches from the RIGHT side to find trailing black pixels,
    avoiding premature cutoff in the middle of the date (which happens with 2-digit hours).
    """
    frame = frame[:, REMOVE:]
    date = frame[DATETOP:DATEBOTTOM, DATELEFT:DATERIGHT]
    
    h, w, _ = date.shape
    
    # Vectorized pixel counting
    col_sums = np.sum(date, axis=2)
    black_counts = np.sum(col_sums < 100, axis=0)
    
    # FIXED: Search from the RIGHT side instead of left
    # This finds the trailing black region (background) rather than
    # accidentally cutting at gaps between date components
    stop_idx = w
    stop = 0
    for x in range(w - 1, -1, -1):  # Iterate from right to left
        if black_counts[x] >= 15:
            stop += 1
        else:
            stop = 0
        if stop > 7:
            stop_idx = x + 8  # Keep the last non-black column
            break
    
    date = date[:, :stop_idx]
    return date


def get_date_fixed(frame, lang='en'):
    """
    FIXED version of get_date that uses right-to-left search for background removal.
    """
    frame = frame[:, REMOVE:]
    date = frame[DATETOP:DATEBOTTOM, DATELEFT:DATERIGHT]
    
    h, w, _ = date.shape
    col_sums = np.sum(date, axis=2)
    black_counts = np.sum(col_sums < 100, axis=0)
    
    # FIXED: Search from the RIGHT side
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
    
    if lang == 'en':
        indices = np.delete(indices, [-5, -6, -1])
    elif lang == 'eu':
        indices = np.delete(indices, [-3, -4])
    
    stats = stats[1:]
    letters = []
    for i in range(len(indices)):
        ind = indices[i]
        if i == len(indices) - 2:
            stats[ind][2] += 1
        letters.append(isolate_letter(date, stats[ind], [18, 12]))
    
    date_str, slash_ind, slashes = get_date_helper(letters, lang=lang)
    if slashes < 2:
        date_str, slash_ind, slashes = get_date_helper(letters, slash=True, lang=lang)
    
    if lang == 'en':
        result = np.sum(letters[-1][:, 6:])
        if result < 24300:
            date_str += 'PM'
        else:
            date_str += 'AM'
    
    date_str = date_str[:slash_ind+5] + ' ' + date_str[slash_ind+5:]
    
    if lang == 'eu':
        day, month, year = date_str.split('/')
        return f"{month}/{day}/{year}"
    
    return date_str


def analyze_date_components(frame, lang='en'):
    """Analyze the connected components in the date region."""
    date_region = extract_date_region(frame, lang)
    
    gray = cv2.cvtColor(date_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Sort by x position (excluding background at index 0)
    indices = np.argsort(centroids[1:, 0])
    
    print(f"\n{'='*60}")
    print("Date Component Analysis")
    print(f"{'='*60}")
    print(f"Total components (excluding background): {num_labels - 1}")
    print(f"\nComponent details (sorted by x position):")
    print(f"{'Index':>6} {'X':>8} {'Y':>8} {'Width':>8} {'Height':>8} {'Area':>8}")
    print("-" * 54)
    
    for i, idx in enumerate(indices):
        real_idx = idx + 1  # +1 because we excluded background
        s = stats[real_idx]
        c = centroids[real_idx]
        print(f"{i:>6} {c[0]:>8.1f} {c[1]:>8.1f} {s[2]:>8} {s[3]:>8} {s[4]:>8}")
    
    print(f"\n{'='*60}")
    
    # For English: indices to remove are [-5, -6, -1]
    # This removes: colon dots (2 components) and 'M'
    # Expected remaining: mm/dd/yyyy + 1-2 hour digits + 2 minute digits + A/P
    
    if lang == 'en':
        print(f"\nFor English, removing indices: [-5, -6, -1] (colon dots and M)")
        print(f"That means removing components at positions: {len(indices)-5}, {len(indices)-6}, {len(indices)-1}")
        expected_single_hour = 14  # mm/dd/yyyy (8) + / (2) + 1 (hour) + 2 (min) + AM/PM (2) - we don't count colon dots
        expected_double_hour = 15  # mm/dd/yyyy (8) + / (2) + 2 (hour) + 2 (min) + AM/PM (2) - we don't count colon dots
        print(f"Expected components for single-digit hour: ~{expected_single_hour}")
        print(f"Expected components for 2-digit hour: ~{expected_double_hour}")
    
    return num_labels - 1, indices


def test_date_parsing_on_frame(frame, expected_date=None, lang='en'):
    """Test date parsing on a specific frame and show detailed debug info."""
    print(f"\n{'='*60}")
    print("Date Parsing Test")
    print(f"{'='*60}")
    
    # Analyze components
    num_components, indices = analyze_date_components(frame, lang)
    
    # Get the actual date
    try:
        result = get_date(frame, lang)
        print(f"\nParsed date: {result}")
        
        if expected_date:
            if result == expected_date:
                print(f"✓ MATCH: Expected '{expected_date}'")
            else:
                print(f"✗ MISMATCH:")
                print(f"  Expected: '{expected_date}'")
                print(f"  Got:      '{result}'")
    except Exception as e:
        print(f"\nError parsing date: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def test_video_first_frames(video_path, num_frames=10, lang='en'):
    """Test date parsing on the first N frames of a video."""
    print(f"\n{'='*60}")
    print(f"Testing first {num_frames} frames of video")
    print(f"Video: {video_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    cam = cv2.VideoCapture(video_path)
    
    frame_results = []
    for i in range(num_frames):
        ret, frame = cam.read()
        if not ret:
            print(f"Could not read frame {i}")
            break
        
        print(f"\n{'='*60}")
        print(f"FRAME {i}")
        print(f"{'='*60}")
        
        try:
            # Resize if needed (1080p to 720p)
            if frame.shape == (1080, 1920, 3):
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            
            result = test_date_parsing_on_frame(frame, lang=lang)
            frame_results.append((i, result, None))
        except Exception as e:
            print(f"Error on frame {i}: {e}")
            frame_results.append((i, None, str(e)))
    
    cam.release()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for frame_num, result, error in frame_results:
        if error:
            print(f"Frame {frame_num}: ERROR - {error}")
        else:
            print(f"Frame {frame_num}: {result}")
    
    return frame_results


def test_letter_isolation_debug(frame, lang='en'):
    """Debug the letter isolation process for a frame."""
    from vid2data import REMOVE, DATELEFT, DATERIGHT, DATETOP, DATEBOTTOM
    
    frame = frame[:, REMOVE:]
    date = frame[DATETOP:DATEBOTTOM, DATELEFT:DATERIGHT]
    
    h, w, _ = date.shape
    col_sums = np.sum(date, axis=2)
    black_counts = np.sum(col_sums < 100, axis=0)
    
    stop_idx = w
    stop = 0
    for x in range(w):
        if black_counts[x] >= 15:
            stop += 1
        else:
            stop = 0
        if stop > 7:
            stop_idx = x
            break
    date = date[:, :stop_idx]
    
    gray = cv2.cvtColor(date, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    indices = np.argsort(centroids[1:, 0])
    
    print(f"\nTotal sorted components: {len(indices)}")
    
    if lang == 'en':
        # Remove colon components and M
        indices_to_remove = [-5, -6, -1]
        print(f"Removing indices: {indices_to_remove}")
        print(f"Which correspond to positions: {[len(indices) + i for i in indices_to_remove]}")
        
        # Show what we're keeping
        kept_indices = [i for i in range(len(indices)) if i not in [len(indices) + x for x in indices_to_remove]]
        print(f"Keeping {len(kept_indices)} components")
        
        try:
            indices = np.delete(indices, [-5, -6, -1])
        except IndexError as e:
            print(f"ERROR during index deletion: {e}")
            print(f"Array length was: {len(indices)}, can't delete those indices")
            return
    
    print(f"\nAfter deletion, {len(indices)} components remain")
    
    stats = stats[1:]
    letters = []
    
    print(f"\nIsolating {len(indices)} letters:")
    for i in range(len(indices)):
        ind = indices[i]
        print(f"  Letter {i}: stats index {ind}, pos ({stats[ind][0]}, {stats[ind][1]}), "
              f"size {stats[ind][2]}x{stats[ind][3]}")
        if i == len(indices) - 2:
            stats[ind][2] += 1
        letters.append(isolate_letter(date, stats[ind], [18, 12]))
    
    print(f"\nTotal letters isolated: {len(letters)}")
    
    # For English: colon insertion position
    if lang == 'en':
        colon_position = len(letters) - 4
        print(f"Colon would be inserted at position {colon_position}")
        print(f"Format should be: MM/DD/YYYY H:MM or MM/DD/YYYY HH:MM")
        print(f"With {len(letters)} letters (excluding M), structure:")
        print(f"  Positions 0-1: Month")
        print(f"  Position 2: /")
        print(f"  Positions 3-4: Day")
        print(f"  Position 5: /")
        print(f"  Positions 6-9: Year")
        print(f"  Positions 10+: Hour")
        print(f"  Colon at: {colon_position}")
        print(f"  After colon: Minutes (2 digits) + AM/PM letter (1)")
    
    return letters


def save_debug_images(frame, output_dir='debug_output', lang='en'):
    """Save debug images showing the date region and components."""
    os.makedirs(output_dir, exist_ok=True)
    
    date_region = extract_date_region(frame, lang)
    cv2.imwrite(f'{output_dir}/date_region.png', date_region)
    
    gray = cv2.cvtColor(date_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{output_dir}/date_binary.png', binary)
    
    # Draw components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    debug_img = cv2.cvtColor(date_region.copy(), cv2.COLOR_BGR2RGB)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(debug_img, str(i-1), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    cv2.imwrite(f'{output_dir}/date_components.png', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    print(f"Debug images saved to {output_dir}/")


def main():
    """Main test function."""
    # Test the specific video mentioned
    video_filename = "20250722-01K0TKVF7Z78ZJAC32A2S97DT3-2A1FC1DA-BD9B-4895-A912-1BFB99244A8A - :❨.mov"
    video_path = f"videos/{video_filename}"
    
    # Replace the special character if needed
    video_filename_alt = video_filename.replace(':', '/')
    video_path_alt = f"videos/{video_filename_alt}"
    
    # Create debug output directory
    os.makedirs('debug_output', exist_ok=True)
    
    # Find the video
    actual_path = None
    if os.path.exists(video_path):
        actual_path = video_path
    elif os.path.exists(video_path_alt):
        actual_path = video_path_alt
    else:
        videos_dir = 'videos'
        if os.path.exists(videos_dir):
            for f in os.listdir(videos_dir):
                if '20250722-01K0TKVF7Z78ZJAC32A2S97DT3' in f:
                    actual_path = os.path.join(videos_dir, f)
                    print(f"Found: {f}")
                    break
    
    if actual_path is None:
        print("Video not found")
        return
    
    print("="*70)
    print("TESTING ORIGINAL vs FIXED DATE PARSING")
    print("="*70)
    print(f"\nVideo: {actual_path}")
    print("\nExpected date from raw image: 7/4/2025 10:23PM (2-digit hour)")
    print()
    
    # Extract and test first few frames
    cam = cv2.VideoCapture(actual_path)
    
    results = []
    for i in range(3):
        ret, frame = cam.read()
        if not ret:
            break
        
        # Resize if 1080p
        if frame.shape == (1080, 1920, 3):
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        
        # Save debug images
        cv2.imwrite(f'debug_output/frame_{i}_full.png', frame)
        
        info = frame[:, REMOVE:]
        date_raw = info[DATETOP:DATEBOTTOM, DATELEFT:DATERIGHT]
        cv2.imwrite(f'debug_output/frame_{i}_date_raw.png', date_raw)
        
        # Compare original vs fixed
        date_original = extract_date_region(frame)
        date_fixed = extract_date_region_fixed(frame)
        
        cv2.imwrite(f'debug_output/frame_{i}_date_original.png', date_original)
        cv2.imwrite(f'debug_output/frame_{i}_date_fixed.png', date_fixed)
        
        print(f"Frame {i}:")
        print(f"  Raw region width:      {date_raw.shape[1]} px")
        print(f"  Original cutoff width: {date_original.shape[1]} px  <- BUG: cuts at space between year and hour")
        print(f"  Fixed cutoff width:    {date_fixed.shape[1]} px  <- FIXED: searches from right side")
        
        # Test the actual date parsing
        try:
            original_result = get_date(frame, 'en')
        except Exception as e:
            original_result = f"ERROR: {e}"
        
        try:
            fixed_result = get_date_fixed(frame, 'en')
        except Exception as e:
            fixed_result = f"ERROR: {e}"
        
        print(f"  Original parsed date: '{original_result}'")
        print(f"  Fixed parsed date:    '{fixed_result}'")
        print()
        
        results.append({
            'frame': i,
            'original': original_result,
            'fixed': fixed_result,
            'original_width': date_original.shape[1],
            'fixed_width': date_fixed.shape[1]
        })
    
    cam.release()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Frame':>6} | {'Original':>25} | {'Fixed':>25}")
    print("-" * 62)
    for r in results:
        orig = r['original'][:25] if len(r['original']) <= 25 else r['original'][:22] + "..."
        fixed = r['fixed'][:25] if len(r['fixed']) <= 25 else r['fixed'][:22] + "..."
        print(f"{r['frame']:>6} | {orig:>25} | {fixed:>25}")
    
    print("\n" + "="*70)
    print("ROOT CAUSE ANALYSIS")
    print("="*70)
    print("""
The bug occurs because the get_date function uses a left-to-right search
to find 7+ consecutive columns with ≥15 black pixels for background removal.

With 2-digit hours (10:XX, 11:XX, 12:XX), the space between the year (2025)
and the time creates a gap that satisfies this condition, causing premature
cutoff at ~90 pixels instead of keeping the full ~165 pixel date region.

THE FIX: Search from the RIGHT side instead to find the trailing background,
rather than accidentally cutting at gaps between date parts.
""")
    print("Debug images saved to debug_output/")


if __name__ == '__main__':
    main()
