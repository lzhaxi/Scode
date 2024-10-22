import os
import shutil
import cv2
import argparse
import subprocess

# was lazy to implement pictures separately, so this makes a video of the image and runs vid2data

import vid2data as v

# You must have at least 2 images in 'pictures' folder for this to work!
# This is only intended as used for non-random rotations
def main(filename='video_from_images.mp4', lang='en', print_to_file=False, rand=False):
    images = [img for img in os.listdir('pictures') if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(images) == 0:
        print("Error: no images in /pictures directory. Note that only .png and .jpg/.jpeg files are supported.")
        return
    if rand:
        if len(images) != 2:
            print("Error: with random scenarios, only include the two screenshots that correspond with each other. Exiting...")
            return
        # check for the scenario being correct, where waves are first and weapons are second
    for i in range(len(images)-1, -1, -1):
        image_path = os.path.join('pictures', images[i])
        frame = cv2.imread(image_path)
        if frame.shape != (720, 1280, 3):
            if rand:
                print('Error: one or more of the images is not 720p. Cannot run due to rand flag')
                return
            print('Image at ' + image_path + ' is not 720p. Skipping..')
            del images[i]

    images.sort()
    with open('image_file_list.txt', 'w') as f:
        for image in images:
            image_path = os.path.join('pictures', image)
            f.write(f"file '{image_path}'\n")

    if not '.' in filename:
        filename += '.mp4'
    path = v.new_path('videos/' + filename)
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',              # Disable filename checking
        '-i', 'image_file_list.txt',
        '-framerate', '30',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuvj420p',
        path
    ]
    try:
        # Run the ffmpeg command
        subprocess.run(command, check=True)
        print(f"Video successfully created: {path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary file list
        if os.path.exists('image_file_list.txt'):
            os.remove('image_file_list.txt')

    for image in images:
        image_path = os.path.join('pictures', image)
        shutil.move(image_path, v.new_path(os.path.join('pictures/done', image)))

    v.main(path[7:], lang, print_to_file, rand or len(images) == 1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts all scenario pictures in /pictures to data to be uploaded to the main google sheet or saved as a local csv.")
    parser.add_argument('img', type=str, default="video_from_images.mp4", nargs='?', help='New filename of scenario video to be made, defaults to "video_from_images.mp4"')
    parser.add_argument('--lang', '-l', type=str, default='en', help='Language, options are \'en\' or \'eu\'')
    parser.add_argument('--print', '-p', help="Prints to csv instead of uploading to google sheets", action='store_true')
    parser.add_argument('--rand', '-r', help='Whether images are from rotation with random weapons (optional)', action='store_true')
    args = parser.parse_args()
    main(args.img, args.lang, args.print, args.rand)