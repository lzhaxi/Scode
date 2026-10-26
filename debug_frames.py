import cv2
import vid2data as v

filename = 'videos/test/2025091912100201_s.mp4'
cam = cv2.VideoCapture(filename)

# pass 1 dummy run
ret, nextFrame = cam.read()
if nextFrame.shape in v.SUPPORTED_RESOLUTIONS and v.SUPPORTED_RESOLUTIONS[nextFrame.shape] is not None:
    nextFrame = v.resize_frame(nextFrame)

date = v.get_date(nextFrame, 'eu')
currentFrame = -1
flagEqual = False
equal = 0
flagInit = True
toRemove = []
row = 3
result_arr = []

while ret and not flagEqual:
    frame = nextFrame
    date_val = date
    ret, nextFrame = cam.read()
    if ret and nextFrame.shape in v.SUPPORTED_RESOLUTIONS and v.SUPPORTED_RESOLUTIONS[nextFrame.shape] is not None:
        nextFrame = v.resize_frame(nextFrame)
    currentFrame += 1
    if ret:
        date = v.get_date(nextFrame, 'eu')
    
    try:
        img_diff = v.compare_image_diff(frame, nextFrame)
        if img_diff or date_val == date:
            frames_same = True
        else:
            frames_same = False
    except v.BlackCodeException:
        if date_val != date:
            frames_same = False
        else:
            frames_same = True
            
    if frames_same:
        if not flagInit:
            equal += 1
        if equal < 10:
            continue
        else:
            flagEqual = True
            
    equal = 0
    if '-' in date_val:
        continue
    
    # Just save the row
    dateFile = date_val.replace(':', ' ').replace('/', ' ') + '.png'
    toRemove.append([dateFile, currentFrame, False, 0, row, False, False])
    result_arr.append([])
    row += 1
    flagInit = False

print(f"Pass 1 ended at frame: {currentFrame}")
print(f"Number of scenarios found: {len(result_arr)}")

# Now simulate what the script does
test_cam = cv2.VideoCapture(filename)
test_cam.set(cv2.CAP_PROP_POS_FRAMES, currentFrame + 1)
frames_checked = 0
video_continues = False

while frames_checked < 600:
    ret_test, frame_test = test_cam.read()
    if not ret_test:
        print(f"Reached end of video after checking {frames_checked} frames.")
        break
        
    if frame_test.shape in v.SUPPORTED_RESOLUTIONS and v.SUPPORTED_RESOLUTIONS[frame_test.shape] is not None:
        frame_test = v.resize_frame(frame_test)
    
    try:
        diff = v.compare_image_diff(frame, frame_test)
    except v.BlackCodeException:
        diff = True
        
    if not diff:
        video_continues = True
        print(f"Found DIFFERENT frame after checking {frames_checked + 1} frames.")
        break
        
    frames_checked += 1

test_cam.release()
print(f"video_continues: {video_continues} after {frames_checked + 1} frames")
