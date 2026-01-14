import os
import shutil
import numpy as np
import csv
import warnings
import time
import socket

import pygsheets
from pydrive.auth import GoogleAuth
from pydrive.auth import RefreshError
from pydrive.drive import GoogleDrive

import skimage as sk
import cv2
import pytesseract
import argparse
from Levenshtein import distance as levenshtein

# constants

FOLDERID='1rz8aL7KwH6Ifiw1LmYRvrjMbvtFj_C98'

# Supported video resolutions: maps input shape to target resize dimensions (or None if no resize needed)
SUPPORTED_RESOLUTIONS = {
    (720, 1280, 3): None,      # 720p - no resize needed (Switch 1)
    (1080, 1920, 3): (720, 1280),  # 1080p - resize to 720p (Switch 2)
}

def resize_frame(frame):
    """
    Resize frame to 720p if it's a supported higher resolution.
    Returns the frame unchanged if already 720p.
    """
    shape = frame.shape
    if shape not in SUPPORTED_RESOLUTIONS:
        return None  # Unsupported resolution
    target = SUPPORTED_RESOLUTIONS[shape]
    if target is None:
        return frame  # Already 720p
    # Resize using INTER_AREA for best quality when downscaling
    return cv2.resize(frame, (target[1], target[0]), interpolation=cv2.INTER_AREA)

# Image cache to avoid repeated disk reads
_IMAGE_CACHE = {}

def get_cached_image(path):
    """Load an image from disk, caching it for subsequent accesses."""
    if path not in _IMAGE_CACHE:
        _IMAGE_CACHE[path] = cv2.imread(path)
    return _IMAGE_CACHE[path]

# Pre-compute file listings to avoid repeated os.listdir calls
ROTATION_IMAGES = [f for f in os.listdir('ground_truths/rotation-images') if f.endswith('.png')]
WEAPON_IMAGES = [f for f in os.listdir('ground_truths/weapon-images') if f.endswith('.png')]
ROLLER_IMAGES = [f for f in os.listdir('ground_truths/weapon-images/Rollers') if f.endswith('.png')]
SWORD_IMAGES = [f for f in os.listdir('ground_truths/weapon-images/Swords') if f.endswith('.png')]
LANG=['en', 'eu', 'jp']
# eu option is DD/MM/YYYY with the date, in English, with no other differences

# pixel locations
REMOVE = 620

##subsequent pixel coordinates are after removal
# weapon locations
WRIGHT = 379
WWIDTH = 39
WTOP = 320
WHEIGHT = 35 # height of the weapon itself
WHEIGHT2 = 24 # distance between successive weapons

# text locations after removal
CODELEFT = 356
CODERIGHT = 627
CODETOP = 596
CODEBOTTOM = 638

# hazard level locations
HAZLEFT = 11
HAZRIGHT = 275
HAZTOP = 105
HAZBOTTOM = 137
HAZLANG = {'en': 178, 'eu': 178, 'jp': 0}
# jp value is a placeholder

# rotation weapon locations
RLEFT = 62
RWIDTH = 44
RTOP = 168
RBOTTOM = 208

# map locations
MAPLEFT = 14
MAPRIGHT = 200
MAPTOP = 72
MAPBOTTOM = 97

# date locations
DATELEFT = 17
DATERIGHT = 200
DATETOP = 46
DATEBOTTOM = 67

# wave checkmark locations
WAVELEFT = 549
WAVERIGHT = 579
WAVETOP = 313
WAVEBOTTOM = 539

# wave type locations
WAVEWIDTH = 158  # mudmouth eruptions longest wave
WAVEHEIGHT = 43  # height of the wave itself
WAVEHEIGHT2 = 62 # distance between successive waves
WAVETYPELEFT = 299
WAVETYPETOP = (408, 374, 342, 309) # indices for the location of the first wave given number of waves
#king name locations
KINGTOP = 501
KINGBOTTOM = 526
KINGLEFT = 184
KINGRIGHT = 290

GREENROUND = {'en': 100, 'eu': 90, 'jp': 1}
# jp value is a placeholder

class BlackCodeException(Exception):
    def __init__(self, message="The video contains a code that has been missed by the nintendo switch capture software. Please manually remove the frame or frames from the video and re-run the script."):
        self.message = message
        super().__init__(self.message)

def leven(tess, key, lang='en'):
    """
    Using levenshtein distance, returns the key that is the closest to the tesseract output string.

    Parameters:
        tess: pytesseract output string
        key: the key for all ground truth possibilities
        lang: the language to look for in the key

    Returns:
        The key that is most likely given it is the closest in distance
    """
    distances = np.ones(len(key)) * 100
    for i in range(len(key)):
        keyword = key[i][lang]
        if tess == keyword:
            return keyword
        distances[i] = levenshtein(tess, keyword)
    return key[np.argmin(distances)][lang]

def levens(tess, key, lang='en'):
    """
    Similar as above but returns distance as well as key.

    Parameters:
        tess: pytesseract output string
        key: the key for all ground truth possibilities
        lang: the language to look for in the key

    Returns:
        The key that is most likely given it is the closest in distance
    """
    distances = np.ones(len(key)) * 100
    for i in range(len(key)):
        keyword = key[i][lang]
        if tess == keyword:
            return keyword, 0
        distances[i] = levenshtein(tess, keyword)
    return key[np.argmin(distances)][lang], np.min(distances)

def ssim(img, img2):
    """Gets similarity of two images with scikit-image, matching implementation of https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf"""
    return sk.metrics.structural_similarity(img, img2, data_range=255, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

def isolate_letter(image, stats, size):
    """
    Used to pad individual images of digits for ssim to compare images in get_hazard.

    Parameters:
        image: the individual image of 1 digit
        stats: the coordinates and width/height of the ROI

    Returns:
        Image padded with 0s on all sides
    """
    x, y, w, h, _ = stats
    y1, x1 = size

    pad_x = (x1 - w) // 2
    pad_y = (y1 - h) // 2
    pad_x1 = pad_x + 1 if (x1 - w) % 2 == 1 else pad_x
    pad_y1 = pad_y + 1 if (y1 - h) % 2 == 1 else pad_y
    return np.pad(image[y:y+h, x:x+w], ((pad_y, pad_y1), (pad_x1, pad_x), (0, 0)), mode='constant', constant_values=0)

def detect_text(img, config='--psm 7'):
    """
    Get text from image using pytesseract.
    psm options:
        6    Assume a single uniform block of text.
        7    Treat the image as a single text line.

    Parameters:
        img: Image to detect text from
        config: Configuration parameters for tesseract

    Returns:
        Predicted string
    """
    result = pytesseract.image_to_string(img, config=config)
    return result.strip()

def get_date(frame, lang='en'):
    """Gets date from image."""
    frame = frame[:, REMOVE:]
    date = frame[DATETOP:DATEBOTTOM, DATELEFT:DATERIGHT]
    # remove the background so it doesn't interfere
    # counts the columns that have majority black pixels, and the image can be cut off at the end where there are many columns with black pixels in a row
    h, w, _ = date.shape
    
    # Vectorized pixel counting: sum across color channels, then count black pixels per column
    col_sums = np.sum(date, axis=2)  # shape: (h, w) - sum of RGB values per pixel
    black_counts = np.sum(col_sums < 100, axis=0)  # count of black pixels per column
    
    # Find trailing black region by searching from RIGHT to LEFT
    # This avoids premature cutoff at gaps between date components (e.g., between year and 2-digit hours like "10:23")
    stop_idx = w
    stop = 0
    for x in range(w - 1, -1, -1):  # Search from right to left
        if black_counts[x] >= 15:
            stop += 1
        else:
            stop = 0
        if stop > 7:
            stop_idx = x + 8  # Keep everything up to the start of the trailing black region
            break
    date = date[:, :stop_idx]
    # assume last two letters are PM or AM, separate the two parts and use detect_text with different config
    gray = cv2.cvtColor(date, cv2.COLOR_BGR2GRAY)
    # unsure if binary or otsu is better
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    indices = np.argsort(centroids[1:, 0])
    if lang == 'en':
        # the colon has two separate components, remove both as well as the 'M'
        indices = np.delete(indices, [-5, -6, -1])
    elif lang == 'eu':
        # only remove the colon, there is no PM or AM
        indices = np.delete(indices, [-3, -4])
    stats = stats[1:]
    letters = []
    for i in range(len(indices)):
        # (18, 12) these values obtained from looking at 1 47 2083PM.png maximum letter size, pad value 3
        ind = indices[i]
        if i == len(indices) - 2:
            stats[ind][2] += 1 # the last digit is cut off
        letters.append(isolate_letter(date, stats[ind], [18, 12]))
    date, slash_ind, slashes = get_date_helper(letters, lang=lang)
    if slashes < 2:
        date, slash_ind, slashes = get_date_helper(letters, slash=True, lang=lang)
    if lang == 'en':
        # Use top/bottom ratio to distinguish A from P
        # A is symmetric (ratio ~1.12-1.14), P has bowl at top only (ratio ~1.44)
        last_letter = letters[-1]
        h = last_letter.shape[0]
        top_sum = np.sum(last_letter[:h//2, :])
        bottom_sum = np.sum(last_letter[h//2:, :])
        if bottom_sum > 0 and (top_sum / bottom_sum) > 1.25:
            date += 'PM'
        else:
            date += 'AM'
    # the space is after the 4 digits of the year after the second slash
    date = date[:slash_ind+5] + ' ' + date[slash_ind+5:]
    if lang == 'eu':
        day, month, year = date.split('/')
        # Return the formatted string in mm/dd/yyyy
        return f"{month}/{day}/{year}"
    return date

def get_date_helper(letters, slash=False, lang='en'):
    # go through each letter and compare each letter to the ground truth data and choose the one that is most similar
    date = ''
    letter_list = ['0', '02', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'slash']
    slash_ind = 0 # will be used to find where the space should be inserted into the date
    slashes = 0 # number of slashes found, this is a second step to ensure the slash is not misidentified as a 1 or 7
    if lang == 'en':
        # don't need the M at the end
        length = len(letters) - 1
    elif lang == 'eu':
        length = len(letters)
    for i in range(length):
        results = []
        for letter in letter_list:
            letter_base = get_cached_image('ground_truths/letter_data/' + letter + '.png')
            result = ssim(letters[i], letter_base)
            results.append(result)
        result = letter_list[np.argmax(results)]
        # code has a hard time recognizing certain digits, so I have multiple examples
        if result == '9' or result == '4':
            four = ssim(letters[i], get_cached_image('ground_truths/letter_data/42.png'))
            four2 = ssim(letters[i], get_cached_image('ground_truths/letter_data/43.png'))
            nine = ssim(letters[i], get_cached_image('ground_truths/letter_data/92.png'))
            nine2 = ssim(letters[i], get_cached_image('ground_truths/letter_data/93.png'))
            let_list = [result, '4', '4', '9', '9']
            result_list = [np.max(results), four, four2, nine, nine2]
            result = let_list[np.argmax(result_list)]
        if result == '5' or result == '6' or result == '8':
            five = ssim(letters[i], get_cached_image('ground_truths/letter_data/52.png'))
            six = ssim(letters[i], get_cached_image('ground_truths/letter_data/62.png'))
            eight = ssim(letters[i], get_cached_image('ground_truths/letter_data/8.png'))
            let_list = [result, '5', '6', '8']
            result_list = [np.max(results), five, six, eight]
            result = let_list[np.argmax(result_list)]
        if result == '1' or result == '7':
            seven1 = ssim(letters[i], get_cached_image('ground_truths/letter_data/72.png'))
            seven2 = ssim(letters[i], get_cached_image('ground_truths/letter_data/73.png'))
            one = ssim(letters[i], get_cached_image('ground_truths/letter_data/12.png'))
            one2 = ssim(letters[i], get_cached_image('ground_truths/letter_data/13.png'))
            let_list = [result, '1', '1', '7', '7']
            result_list = [np.max(results), one, one2, seven1, seven2]
            if slash:
                let_list.append('slash')
                slash1 = ssim(letters[i], get_cached_image('ground_truths/letter_data/slash1.png'))
                result_list.append(slash1)
            result = let_list[np.argmax(result_list)]
        # the filename can't contain a slash, so this is the workaround
        if result == 'slash':
            result = '/'
            slash_ind = i
            slashes += 1
        if len(result) > 1:
            result = result[0]
        date += result
        if lang == 'en':
            if i == len(letters) - 4:
                date += ':'
        elif lang == 'eu':
            if i == len(letters) - 3:
                date += ':'
    return date, slash_ind, slashes


def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def get_num_waves(info, template, lang='en'):
    """Get the number of waves based on how many times a checkmark or x appears"""
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]

    wave_img = cv2.cvtColor(info[WAVETOP:WAVEBOTTOM, WAVELEFT:WAVERIGHT], cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(wave_img, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= 0.8) #type: ignore
    locations_list = list(zip(*locations[::-1]))

    min_dist = max(w, h) // 2
    filtered = []
    # remove all redundant locations, determined by min_dist
    for loc in locations_list:
        if all(euclidean_distance(loc, other_loc) > min_dist for other_loc in filtered):
            filtered.append(loc)
    return len(filtered)

def get_hazard(info, lang='en'):
    """
    Gets hazard from image.
    lang is needed because the position of the hazard differs depending on language.
    Uses hazard_data to compare images using ssim
    """
    # method using pytesseract instead
    #hazard = detect_text(info[HAZTOP:HAZBOTTOM, HAZLEFT:HAZRIGHT], '--psm 7 -c tessedit_char_whitelist=0123456789')

    threshold = 0.7 # determined through testing
    haz = info[HAZTOP:HAZBOTTOM, HAZLEFT+HAZLANG[lang]:HAZRIGHT]
    # binarize image for cv2.connectedComponentsWithStats
    #binary = process_image(haz)
    gray = cv2.cvtColor(haz, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    # remove components that are too short, which are sometimes erroneously picked up
    length = len(stats)
    for i in range(length):
        if i >= length:
            break
        if stats[i][3] < 10:
            stats = np.delete(stats, i, axis=0)
            centroids = np.delete(centroids, i, axis=0)
            length = length - 1

    # remove the percent symbol, which is at the max x value
    max = np.argmax(centroids[:, 0])
    stats = np.delete(stats, max, axis=0)
    centroids = np.delete(centroids, max, axis=0)

    # separate each digit into its own image
    digits = []
    for i in range(1, len(stats)):
        # (24, 21) these values obtained from looking at 1wave.JPG maximum digit size, pad value 5
        digits.append(isolate_letter(haz, stats[i], [24, 21]))
    digits = [x for _, x in sorted(zip(centroids[1:, 0], digits))]
    hazard = ''
    digit_list = '0123456789'

    # go through each digit and compare each digit to the ground truth data and choose the one that is most similar
    for digit_img in digits:
        results = []
        for digit in digit_list:
            digit_base = get_cached_image('ground_truths/hazard_data/' + digit + '.png')
            results.append(ssim(digit_img, digit_base))
        hazard += digit_list[np.argmax(results)]
    # sometimes stats gets another element in its array, delete it and log it
    if len(hazard) > 3:
        #print('Possible error, hazard was ' + hazard + ', image logged. Deleting fourth digit, correct manually if wrong.')
        #base_name = hazard
        #while os.path.exists('out/' + base_name + '.png'):
        #    base_name += "_"
        #cv2.imwrite('out/' + base_name + '.png', info)
        hazard = hazard[:-1]
    return hazard

def get_wave_type(wave, event_key, water_key, lang='en'):
    """
    Gets the wave type based on levenshtein distance.

    Parameters:
        wave: image for text detection
        event_key: key containing night wave names
        water_key: key containing tide level names
        lang: language
    Returns:
        string containing tide level and event if applicable
    """
    wave = cv2.cvtColor(wave, cv2.COLOR_BGR2GRAY)
    tess = detect_text(wave, config='--psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "')
    #print('Raw detected: ', tess)

    spaces = tess.split(' ')
    for i in range(len(spaces)):
        if i >= len(spaces):
            break
        # Remove one letter tokens, in case they exist - pytesseract sometimes detects them erroneously. The smallest number of letters in wave types is 3 (Fog, Low)
        if len(spaces[i]) == 1:
            del spaces[i]
    tess = ' '.join(spaces)
    if '\n' in tess:
        tess = tess.split('\n')
        # sometimes pytesseract adds extra newlines, and those characters should be included
        if len(tess) > 2:
            for i in range(2, len(tess)):
                tess[1] += tess[i]
        # sometimes, pytesseract erroneously splits "High" and "Tide" or other words when there is no night wave
        event, event_dist = levens(tess[1], event_key)
        _, water_dist = levens(tess[1], water_key)
        # if event is closer to a water key rather than an event key, treat the whole text as water rather than event
        if event == 'Tide' or event == 'Day' or water_dist < event_dist:
            return leven(tess[0] + ' ' + tess[1], water_key)
        else:
            water = leven(tess[0], water_key)
            return water + '\n' + event
    else:
        return leven(tess, water_key)
def get_king(king, king_key, lang='en'):
    """
    Gets the king type based on levenshtein distance.

    Parameters:
        king: image for text detection
        king_key: key containing king names
        lang: language
    Returns:
        string containing king name
    """
    # pad to make it easier for pytesseract
    king = np.pad(king, ((5, 5), (15, 5), (0, 0)), mode='constant', constant_values=0)
    tess = detect_text(king, config='--psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "')
    #print('Raw detected: ', tess)
    return leven(tess, king_key)

def read_keys(key_file):
    """Reads keys from csv key file into dictionary"""
    keys = []
    with open(key_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            keys.append(row)
    return keys

def compare_image_diff(frame, nextFrame):
    """Returns whether images are different based on the pixels of the code"""
    info = frame[:, REMOVE:]
    info2 = nextFrame[:, REMOVE:]
    code = info[CODETOP:CODEBOTTOM, CODELEFT:CODERIGHT, :]
    # check if the rare black code exception happens when nintendo switch capture software fails to save the code at all
    gray = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(thresh) == 0:
        raise BlackCodeException()
    code2 = info2[CODETOP:CODEBOTTOM, CODELEFT:CODERIGHT, :]
    sub = cv2.subtract(code, code2)
    # through testing, determined all same frames had sum difference in the thousands and tens of thousands, max 100000
    # all different frames had >400000
    return np.sum(cv2.sumElems(sub)) < 200000

def new_path(path):
    """Returns path that will not overwrite existing file"""
    if os.path.exists(path):
        counter = 0
        path_next, path_ext = path.rsplit('.', 1)
        path_ext = '.' + path_ext
        while os.path.exists(path_next + '_' + str(counter) + path_ext):
            counter += 1
        path = path_next + '_' + str(counter) + path_ext
    return path

def main(filename, lang='en', print_to_file=False, single=False, from_pics = False):
    """
    Parameters:
    filename: name of file to be read
    print_to_file: whether file should be uploaded to sheets or printed to csv
    single: usage with pics2data, whether the video is of a single scenario
    from_pics: comes from pics2data, the green value is slightly lower for some reason when using pics
    """
    socket.setdefaulttimeout(30) # set timeout for socket operations to 10 seconds, to prevent google drive upload file errors
    print('Current file is: ' + filename)
    if not print_to_file:
        # setup for google drive
        print('Authenticating with Google Drive...', end=' ')
        # ignore warning if creds.txt is not there
        warnings.filterwarnings('ignore')
        gauth = GoogleAuth(settings_file='settings.yaml')
        gauth.LoadCredentialsFile("creds.json")
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            try:
                gauth.Refresh()
            except RefreshError as e:
                gauth.LocalWebserverAuth()
        else:
            # Initialize the saved creds
            gauth.Authorize()
        # Save the current credentials to a file
        gauth.SaveCredentialsFile("creds.json")
        drive = GoogleDrive(gauth)
        gc = pygsheets.authorize(service_file='scode.json')
        # Open spreadsheet and then worksheet
        sh = gc.open('Leo\'s codes v3.1')
        wks = sh.worksheet_by_title('Codes')
        print('Done')
    else:
        path = 'out/' + filename[:-4]
        if (os.path.exists(path)):
            print("Error: directory " + path + " already exists. Exiting...")
            return
        os.mkdir('out/' + filename[:-4])

    # Read keys into variables
    event_key = read_keys('ground_truths/keys/event.csv')
    stage_key = read_keys('ground_truths/keys/stage.csv')
    water_key = read_keys('ground_truths/keys/water.csv')
    king_key = read_keys('ground_truths/keys/king.csv')
    #weapon_key = read_keys('ground_truths/keys/weapon.csv')


    cam = cv2.VideoCapture('videos/' + filename)
    ret, nextFrame = cam.read()
    if not ret:
        print('Error: No frame read from video')
        return
    
    # Check if resolution is supported and resize if needed
    if nextFrame.shape not in SUPPORTED_RESOLUTIONS:
        print(f'Error: Video resolution {nextFrame.shape[1]}x{nextFrame.shape[0]} is not supported. Supported: 720p, 1080p')
        return
    needs_resize = SUPPORTED_RESOLUTIONS[nextFrame.shape] is not None
    if needs_resize:
        print(f'Detected {nextFrame.shape[1]}x{nextFrame.shape[0]} video, will resize frames to 720p')
        nextFrame = resize_frame(nextFrame)

    nextDate = get_date(nextFrame, lang)
    equal = 0 # how many frames in a row have been equivalent
    flagInit = True # whether there are any different frames in the whole video
    if single:
        flagInit = False
    flagEqual = False # indicating enough frames have been the same to move on
    flagRand = False # if there is a random rotation in the video
    currentFrame = -1
    toRemove = []
    row = 3
    if not print_to_file:
        # get the first empty row
        cols = wks.get_col(2)
        # remove empty cells from the list
        cols = list(filter(None, cols))
        row = len(cols) + 2
    initRow = row
    result_arr = []
    black_code_recovery = {}  # Track scenarios needing code recovery: dateFile -> result_arr index
    try:
        while(ret and not flagEqual):
            frame = nextFrame
            date = nextDate
            ret, nextFrame = cam.read()
            if ret and needs_resize:
                nextFrame = resize_frame(nextFrame)
            currentFrame += 1
            if ret:
                nextDate = get_date(nextFrame, lang)
            # if reached end of the video, get data from current frame, then break
            if single:
                flagEqual = True
            elif ret:
                # Check if frames are the same - catch BlackCodeException to defer code capture
                try:
                    frames_same = date == nextDate or compare_image_diff(frame, nextFrame)
                except BlackCodeException:
                    # Black code detected - we'll try to recover it on second pass
                    print(f'Black code detected on frame {currentFrame}, will attempt recovery on second pass')
                    frames_same = False  # Treat as different frame to process this scenario
                    # Mark that we need to capture this scenario's code on second pass
                    # (We'll add to black_code_recovery after building dateFile)
                    black_code_on_current_frame = True
                else:
                    black_code_on_current_frame = False
                
                if frames_same:
                    if not flagInit:
                        equal += 1
                    if equal < 10:
                        # when it reaches 10 equivalent frames in a row, time for weapons phase
                        continue
                    else:
                        flagEqual = True
            dateFile = date + '.png'
            # remove these characters for file name
            dateFile = dateFile.replace(':', ' ')
            dateFile = dateFile.replace('/', ' ')
            # check whether file already exists locally, meaning during this run
            # indicates weapons phase has already been reached
            if os.path.isfile('out/' + dateFile):
                print('Probably an error that ' + date + ' file already exists on frame ' + str(currentFrame))
                break

            equal = 0 # different frame, so reset the number of consecutive equivalent frames
            if '-' in date:
                # date containing '-' indicates that the scenario has not been played yet
                continue
            print('Getting data for frame ' + str(currentFrame), end='... ')
            info = frame[:, REMOVE:]
            # get data
            # determine number of waves with the checkmarks indicating whether a wave was passed
            red = get_num_waves(info, 'ground_truths/template_x.png', lang)
            green = get_num_waves(info, 'ground_truths/template_check.png', lang)
            waves = red + green
            # map
            # returns the most likely stage based on distance between ground truth values for the language and the detected text of pytesseract
            map = info[MAPTOP:MAPBOTTOM, MAPLEFT:MAPRIGHT]
            map = np.pad(map, ((0, 0), (10, 10), (0, 0)), mode='edge')
            map = leven(detect_text(map), stage_key)
            # rotation weapons
            print('Getting rotation weapons...', end=' ')
            rot1 = info[RTOP:RBOTTOM, RLEFT:RLEFT+RWIDTH]
            rot2 = info[RTOP:RBOTTOM, RLEFT+RWIDTH:RLEFT+2*RWIDTH]
            rot3 = info[RTOP:RBOTTOM, RLEFT+2*RWIDTH:RLEFT+3*RWIDTH]
            rot4 = info[RTOP:RBOTTOM, RLEFT+3*RWIDTH:RLEFT+4*RWIDTH]
            max = [[0, ''], [0, ''], [0, ''], [0, '']]
            # compare to weapon images
            rand = False
            for rotfilename in ROTATION_IMAGES:
                weap = get_cached_image('ground_truths/rotation-images/' + rotfilename)
                similarity = (ssim(rot1, weap), ssim(rot2, weap), ssim(rot3, weap), ssim(rot4, weap))
                for i in range(0, 4):
                    if similarity[i] > max[i][0]:
                        max[i][0] = similarity[i]
                        max[i][1] = rotfilename[:-4]
            print('Done')
            if max[0][1] == 'Random' or max[0][1] == 'Gold Random':
                # hard to tell apart random and gold question mark, so use color values
                color_sums = cv2.sumElems(rot1)
                if color_sums[2] > color_sums[0]:
                    rots = 'Gold Random'
                else:
                    rots = 'All Random'
                flagRand = True
                rand = True
                rots_filter = '-'
            elif max[3][1] == 'Random': # if the first index is not a question mark but the last one is, the rotation is 1 random
                flagRand = True
                rand = True
                rots = '1 Random + ' + max[0][1] + ', ' + max[1][1] + ', ' + max[2][1]
                rots_filter = '|' + max[0][1] + '||' + max[1][1] + '||' + max[2][1] + '|'
            else:
                rots = max[0][1] + ', ' + max[1][1] + ', ' + max[2][1] + ', ' + max[3][1]
                rots_filter = '|' + max[0][1] + '||' + max[1][1] + '||' + max[2][1] + '||' + max[3][1] + '|'
            # waves
            top = WAVETYPETOP[waves - 1]
            left = WAVETYPELEFT
            wave1 = get_wave_type(info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH], event_key, water_key)
            wave2 = '-'
            wave3 = '-'
            wave4 = '-'
            if waves > 1:
                top = top + WAVEHEIGHT2
                wave2 = get_wave_type(info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH], event_key, water_key)
            if waves > 2:
                top = top + WAVEHEIGHT2
                wave3 = get_wave_type(info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH], event_key, water_key)
            if waves > 3:
                if '\n' in wave3:
                    wave4 = wave3.split('\n')
                    wave4 = wave4[0]
                else:
                    wave4 = wave3
                wave4 += '\n' + get_king(info[KINGTOP:KINGBOTTOM, KINGLEFT:KINGRIGHT], king_key)
            if waves < 4 and red == 1:
                wave4 = '?' # if loss occurred, boss is unknown
            # handling when get_wave_type returns null, which usually means pytesseract got a blank string
            if any('Null' in ele for ele in [wave1, wave2, wave3, wave4]):
                print('Wave type returned Null, please correct manually\nDate: ' + date + ', frame: ' + str(currentFrame))
                cv2.imwrite('out/Null_' + dateFile[:-4] + '_' + str(currentFrame) + '.png', frame)
            # hazard level
            haz = get_hazard(info)

            # Handle code image - skip for black code scenarios (will capture on second pass if random rotations exist)
            if black_code_on_current_frame:
                # Don't save code image yet - mark for recovery (only works if there are random rotations)
                black_code_recovery[dateFile] = len(result_arr)  # Index in result_arr
                toRemove.append([dateFile, currentFrame, rand, waves, row, True])  # True = needs code recovery
                if print_to_file:
                    code_cell = dateFile  # Placeholder, will be updated on recovery
                else:
                    code_cell = 'PENDING_RECOVERY'  # Will be updated on second pass
            else:
                code = info[CODETOP:CODEBOTTOM, CODELEFT:CODERIGHT]
                cv2.imwrite('out/' + dateFile, code)
                toRemove.append([dateFile, currentFrame, rand, waves, row, False])  # False = no recovery needed
                if not print_to_file:
                    for attempt in range(20):
                        # save image of code to google drive
                        file1 = drive.CreateFile({'title' : dateFile, 'parents': [{'id': FOLDERID}]})
                        file1.SetContentFile('out/' + dateFile)
                        file1['mimeType'] = 'image/png'
                        try:
                            file1.Upload()
                            print('Uploaded code image to Google Drive')
                            break
                        except:
                            if attempt < 4:
                                print('Google Drive upload failed, retrying...')
                            else:
                                raise Exception('Error uploading image to Google Drive')
                    code_cell = '=IMAGE("https://drive.google.com/uc?export=view&id=' + file1['id'] + '")'
                else:
                    code_cell = dateFile

            table_row = [map, rots, haz, wave1, wave2, wave3, wave4]
            table_row += ['-', '-', '-', '-'] # if random weapons, will be altered later
            table_row += [code_cell]
            table_row += [date, '', rots_filter, '-', '-', '-', '-'] # notes is empty, other columns are for search filtering, random weapons altered later
            result_arr.append(table_row)
            row += 1
            flagInit = False
            print('Done with ' + date + ' or frame ' + str(currentFrame))
        if not print_to_file:
            endRow = row - 1
            wks.adjust_row_height(initRow, end=endRow, pixel_size=35)
        cam.release()
        if flagInit:
            print('Error: Images are all the same scenario code')
            return

        # Check if there are black codes that need recovery but no random rotations to enable second pass
        if black_code_recovery and not flagRand:
            raise BlackCodeException('Black code detected with no random rotations - unable to recover via second pass')

        # unnecessary to run the rest of the code if there are no random weapons
        if not flagRand:
            if print_to_file:
                with open('out/' + filename[:-4] + '/full_scenarios.csv', 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(result_arr)
                for i in range(0, len(toRemove)):
                    # Only move file if it exists (won't exist for black code scenarios)
                    if os.path.isfile('out/' + toRemove[i][0]):
                        shutil.move('out/' + toRemove[i][0], 'out/' + filename[:-4] + '/' + toRemove[i][0])
                print('Wrote data to out/' + filename[:-4] + '/full_scenarios.csv')
            else:
                wks.update_values(crange='A' + str(initRow) + ':S' + str(endRow), values=result_arr)
                print('Uploaded data from ' + filename + ' to Leo\'s Codes')
                shutil.move('videos/' + filename, new_path('videos/done/' + filename))
                # only really consider it as done when it is uploaded to the sheet, not just saved locally
                for i in range(0, len(toRemove)):
                    if os.path.isfile('out/' + toRemove[i][0]):
                        os.remove('out/' + toRemove[i][0])
            return

        cam = cv2.VideoCapture('videos/' + filename)
        for i in range(0, toRemove[-1][1]+1): # set frame to first frame of last scenario before weapons phase
            cam.read()
        ret, nextFrame = cam.read()
        if ret and needs_resize:
            nextFrame = resize_frame(nextFrame)
        nextDate = get_date(nextFrame, lang)
        currentFrame = toRemove[-1][1]
        flagInit = False
        flagEqual = False
        equal = 0
        for dateOrigFile, prevFrame, rand, waves, row, needs_recovery in reversed(toRemove):
            if flagEqual:
                break
            dateOrig = dateOrigFile[:-4]
            print('Checking for random weapons in ' + dateOrig + ' or frame ' + str(currentFrame))
            frame = nextFrame
            date = nextDate
            while '-' in date:
                ret, nextFrame = cam.read()
                if ret and needs_resize:
                    nextFrame = resize_frame(nextFrame)
                nextDate = get_date(nextFrame, lang)
                date = nextDate
                currentFrame += 1
            dateFile = date.replace(':', ' ')
            dateFile = dateFile.replace('/', ' ')
            if dateOrig != dateFile:
                print('Possible Error: Frame for wave types is not the same as frame for weapons. Check /out directory for more info.')
                print('dateOrig:' + dateOrig + ', date:' + dateFile)
                file_mode = 'a' if os.path.exists('out/log.txt') else 'w'
                with open('out/log.txt', file_mode) as f:
                    f.write('date: ' + dateFile + ', frame: ' + str(currentFrame) + ', prevFrame: ' + str(prevFrame) + ', row: ' + str(row) + '\n')
                frameNum = '_' + str(prevFrame) + '_' + str(currentFrame) + '_'
                cv2.imwrite('out/frame' + frameNum + filename[:-4] + '.png', frame)
                cam2 = cv2.VideoCapture('videos/' + filename)
                for i in range(0, prevFrame):
                    cam2.read()
                ret, prevF = cam2.read()
                if ret and needs_resize:
                    prevF = resize_frame(prevF)
                cv2.imwrite('out/pFrame' + frameNum + filename[:-4] + '.png', prevF)
                cam2.release()
            ret, nextFrame = cam.read()
            if ret and needs_resize:
                nextFrame = resize_frame(nextFrame)
            currentFrame += 1
            if ret:
                nextDate = get_date(nextFrame, lang)
            # if reached end of the video, get data from current frame, then break
            while ret and (date == nextDate or compare_image_diff(frame, nextFrame)):
                frame = nextFrame
                # check if the frame for wave types is the same as the frame for weapons
                if equal >= 15 and flagInit: # if 15 frames are the same (but not the first 15 frames of the weapons phase) then get the data from current frame and break
                    flagEqual = True
                    break
                ret, nextFrame = cam.read()
                if ret and needs_resize:
                    nextFrame = resize_frame(nextFrame)
                if not ret:
                    break
                nextDate = get_date(nextFrame, lang)
                currentFrame += 1
                equal += 1
            equal = 0
            flagInit = True
            
            # Handle black code recovery - capture code now if needed
            if needs_recovery:
                info = frame[:, REMOVE:, :]
                code = info[CODETOP:CODEBOTTOM, CODELEFT:CODERIGHT]
                # Check if code is still black (extremely rare double black code)
                gray = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                if cv2.countNonZero(thresh) == 0:
                    raise BlackCodeException('Black code detected on BOTH passes - unable to recover')
                
                print(f'Recovered code for {date} on second pass')
                cv2.imwrite('out/' + dateOrigFile, code)
                
                # Update result_arr with code image reference
                ind = row - initRow
                if print_to_file:
                    result_arr[ind][11] = dateOrigFile  # Update the code cell
                else:
                    for attempt in range(20):
                        file1 = drive.CreateFile({'title' : dateOrigFile, 'parents': [{'id': FOLDERID}]})
                        file1.SetContentFile('out/' + dateOrigFile)
                        file1['mimeType'] = 'image/png'
                        try:
                            file1.Upload()
                            print('Uploaded recovered code image to Google Drive')
                            break
                        except:
                            if attempt < 4:
                                print('Google Drive upload failed, retrying...')
                            else:
                                raise Exception('Error uploading image to Google Drive')
                    result_arr[ind][11] = '=IMAGE("https://drive.google.com/uc?export=view&id=' + file1['id'] + '")'
            
            if not rand:
                continue

            print('Getting random weapons for ' + date + '...', end=' ')
            info = frame[:, REMOVE:, :]
            right = WRIGHT
            top = WTOP
            weaps = []
            weaps_filter = []
            for i in range(0, waves):
                weap1 = info[top:top+WHEIGHT, right-WWIDTH:right]
                top = top + WHEIGHT + WHEIGHT2
                weap2 = info[top:top+WHEIGHT, right-WWIDTH:right]
                top = top + WHEIGHT + WHEIGHT2
                weap3 = info[top:top+WHEIGHT, right-WWIDTH:right]
                top = top + WHEIGHT + WHEIGHT2
                weap4 = info[top:top+WHEIGHT, right-WWIDTH:right]
                right = right - WWIDTH
                top = WTOP
                max = [[0, ''], [0, ''], [0, ''], [0, '']]
                for filen in WEAPON_IMAGES:
                    weap = get_cached_image('ground_truths/weapon-images/' + filen)
                    # print shape of both
                    similarity = (ssim(weap1, weap), ssim(weap2, weap), ssim(weap3, weap), ssim(weap4, weap))
                    for j in range(0, 4):
                        if similarity[j] > max[j][0]:
                            max[j][0] = similarity[j]
                            if '_' in filen:
                                # file format is _# attached to the end of the name of the weapon, so remove that
                                max[j][1] = filen[:-6]
                            else:
                                max[j][1] = filen[:-4]
                # Rollers are hard to predict, so without having every weapon always check for multiple images of rollers, only check if predicted weapon is one of the rollers
                for j in range(4):
                    rollers = ['Roller', 'Carbon', 'Dynamo', 'Flingza', 'Swig']
                    if max[j][1] in rollers:
                        match j:
                            case 0:
                                weapx = weap1
                            case 1:
                                weapx = weap2
                            case 2:
                                weapx = weap3
                            case 3:
                                weapx = weap4
                        for filen in ROLLER_IMAGES:
                            weap = get_cached_image('ground_truths/weapon-images/Rollers/' + filen)
                            similarity = ssim(weapx, weap)
                            if similarity > max[j][0]:
                                max[j][0] = similarity
                                max[j][1] = filen[:-6]
                # For some reason, stamper is also hard to predict with GSword. Will check for that only if stamper is the pick
                for j in range(4):
                    swords = ['Stamper']
                    if max[j][1] in swords:
                        match j:
                            case 0:
                                weapx = weap1
                            case 1:
                                weapx = weap2
                            case 2:
                                weapx = weap3
                            case 3:
                                weapx = weap4
                        for filen in SWORD_IMAGES:
                            weap = get_cached_image('ground_truths/weapon-images/Swords/' + filen)
                            similarity = ssim(weapx, weap)
                            if similarity > max[j][0]:
                                max[j][0] = similarity
                                max[j][1] = filen[:-6]
                weaps.append(max[0][1] + ', ' + max[1][1] + ', ' + max[2][1] + ', ' + max[3][1])
                weaps_filter.append('|' + max[0][1] + '||' + max[1][1] + '||' + max[2][1] + '||' + max[3][1] + '|')
            ind = row - initRow
            result_arr[ind][7] = weaps[-1]
            result_arr[ind][15] = weaps_filter[-1]
            if waves > 1:
                result_arr[ind][8] = weaps[-2]
                result_arr[ind][16] = weaps_filter[-2]
            if waves > 2:
                result_arr[ind][9] = weaps[-3]
                result_arr[ind][17] = weaps_filter[-3]
            if waves > 3:
                result_arr[ind][10] = weaps[-4]
                result_arr[ind][18] = weaps_filter[-4]
            print('Done')
        if print_to_file:
            with open('out/' + filename[:-4] + '/full_scenarios.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(result_arr)
            for i in range(0, len(toRemove)):
                if os.path.isfile('out/' + toRemove[i][0]):
                    shutil.move('out/' + toRemove[i][0], 'out/' + filename[:-4] + '/' + toRemove[i][0])
            print('Wrote data to out/' + filename[:-4] + '/full_scenarios.csv')
        else:
            wks.update_values(crange='A' + str(initRow) + ':S' + str(endRow), values=result_arr)
            print('Uploaded data from ' + filename + ' to Leo\'s Codes')
            shutil.move('videos/' + filename, new_path('videos/done/' + filename))
            for i in range(0, len(toRemove)):
                if os.path.isfile('out/' + toRemove[i][0]):
                    os.remove('out/' + toRemove[i][0])
        cam.release()
    except Exception as e:
        print(e)
        cam.release()
        if print_to_file:
            os.removedirs('out/' + filename[:-4])
        for i in range(0, len(toRemove)):
            # Only try to remove files that exist (black code scenarios may not have files)
            if os.path.isfile('out/' + toRemove[i][0]):
                print('Trashing file: '+ toRemove[i][0])
                os.remove('out/' + toRemove[i][0])
                if not print_to_file:
                    # the file definitely exists, but for some reason ListFile sometimes returns empty so you just have to keep querying it
                    while True:
                        file_list = drive.ListFile({'q': f"'{FOLDERID}' in parents and title='{toRemove[i][0]}' and trashed=false"}).GetList()
                        if len(file_list) == 1:
                            break
                        time.sleep(0.5)
                    file = file_list[0]
                    file.Trash()
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts scenario videos to data to be uploaded to the main google sheet or saved as a local csv.",epilog="If no arguments are provided, all the videos in /videos will be uploaded and assumed to be lang=en.")
    parser.add_argument('vids', type=str, help='Filenames of scenario videos. Videos must be in /videos directory', nargs='*')
    parser.add_argument('--lang', '-l', type=str, default='en', help='Language, options are \'en\' or \'eu\'')
    parser.add_argument('--print', '-p', help="Prints to csv instead of uploading to google sheets", action='store_true')
    args = parser.parse_args()
    try:
        if not args.vids:
            flagRan = False
            for filename in os.listdir('videos'):
                if filename.endswith('.mp4'):
                    main(filename, args.lang, args.print)
                    flagRan = True
                if filename.endswith('.mov'):
                    main(filename, args.lang, args.print)
                    flagRan = True
            if not flagRan:
                print('No videos in the directory! Note that only .mov and .mp4 files are supported.')
        else:
            for vid in args.vids:
                if os.path.exists('videos/' + vid):
                    main(vid, args.lang, args.print)
                else:
                    print(vid + ' does not exist. Please check to make sure it has been placed in the /videos directory')
    except BlackCodeException as e:
        print('Stopped running due to black code exception')
    except IndexError as e:
        print('Likely a Job Not Complete, please check the video.')
    except Exception as e:
        print('Error: ' + str(e))
