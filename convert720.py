import ffmpeg
import os
import shutil
import cv2

import vid2data as v

for filename in os.listdir('videos'):
    input_path = os.path.join('videos', filename)
    if filename.endswith(('.mp4', '.mov')):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width != 1920) or (height != 1080):
            print('Resolution of ' + input_path + ' is not 1080p.')
            continue
        namesplit = filename.rsplit('.', 1)
        out_file = namesplit[0] + '_720.' + namesplit[1]
        out_path = os.path.join('videos', out_file)
        print('Resizing ' + filename)
        ffmpeg.input(input_path).filter('scale', 1280, 720).output(out_path).run()
        shutil.move(input_path, v.new_path(os.path.join('videos/done', filename)))
for filename in os.listdir('pictures'):
    input_path = os.path.join('pictures', filename)
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width != 1920) or (height != 1080):
            print('Resolution of ' + input_path + ' is not 1080p.')
            continue
        namesplit = filename.rsplit('.', 1)
        out_file = namesplit[0] + '_720.' + namesplit[1]
        out_path = os.path.join('pictures', out_file)
        print('Resizing ' + filename)
        ffmpeg.input(input_path).filter('scale', 1280, 720).output(out_path).run()
        shutil.move(input_path, v.new_path(os.path.join('pictures/done', filename)))
