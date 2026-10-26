import cv2
import vid2data
filename = 'test/2025120813410901_s.mp4'

cam = cv2.VideoCapture('videos/' + filename)

ret, nextFrame = cam.read()
nextFrame = vid2data.resize_frame(nextFrame)

nextDate = vid2data.get_date(nextFrame, 'en')

equal = 0
flagEqual = False
currentFrame = -1
while ret and not flagEqual:
    frame = nextFrame
    date = nextDate
    ret, nextFrame = cam.read()
    if ret:
        nextFrame = vid2data.resize_frame(nextFrame)
        currentFrame += 1
        nextDate = vid2data.get_date(nextFrame, 'en')
        
        try:
            img_diff = vid2data.compare_image_diff(frame, nextFrame)
            frames_same = (not img_diff) and (date == nextDate)
            bcode = False
        except vid2data.BlackCodeException:
            bcode = True
            frames_same = False
        
        if frames_same:
            equal += 1
            if equal >= 10:
                pass
            continue
        
        print(f"End of scenario at frame {currentFrame}, date={date}, black_code={bcode}")
        equal = 0

cam.release()
