import os
import sys
import time

from google.cloud import vision
import pygsheets
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from PIL import Image
import skimage as sk
import numpy as np
import cv2


# constants

CREDS='/Users/leo/.config/gcloud/application_default_credentials.json'
FOLDERID='1rz8aL7KwH6Ifiw1LmYRvrjMbvtFj_C98'

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
DATETOP = 43
DATEBOTTOM = 70

# wave checkmark locations
WAVELEFT = 549
WAVERIGHT = 579
WAVETOP = 313
WAVEBOTTOM = 539

# wave type locations
WAVEWIDTH = 158  # mudmouth eruptions longest wave
WAVEHEIGHT = 50  # height of the wave itself
WAVEHEIGHT2 = 62 # distance between successive waves
WAVETYPELEFT = 299
WAVETYPETOP = (405, 371, 339, 306) # indices for the location of the first wave given number of waves

# version locations
VLEFT = 52
VRIGHT = 125
VTOP = 602
VBOTTOM = 629

# make dataset for tesstrain after having manually inputted data
def make_gt_files(directory):
    """Makes dataset for tesstrain."""
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            text = filename[:-4]
            textFile = open(directory + '/' + filename[:-4] + '.gt.txt', 'w')
            textFile.write(text)
            textFile.close()

def image_prettify(img, iter=1):
    """Image manipulation for tesseract."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert image
    img = cv2.bitwise_not(img)

    # Apply threshold to get image with only black and white
    ret,img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
     ## Apply erosion to remove some noise
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=iter, borderType=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return img

# get text from image
#def get_string(img):
#    result = ''

#    #Recognize text with tesseract for python
#    result = pytesseract.image_to_string(img, config = '--psm 6, -c tessedit_char_whitelist=-0123456789ABCDEFGHJKLMNPQRSTUVWXY, -c load_system_dawg=0, -c load_freq_dawg=0, -c language_model_penalty_spacing=0.5')

def detect_text(content):
    """Detects text in the image."""
    # convert content to bytes with cv2
    content = cv2.imencode('.png', content)[1].tobytes()
    #with io.open(path, 'rb') as image_file:
    #    content = image_file.read()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=CREDS
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    time.sleep(0.01)
    response = client.text_detection(image=image) #type: ignore
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    texts = response.text_annotations
    # convert all cyrillic characters to latin
    
    return texts[0].description

#def get_digits(img):
#    result = ''

#    result = pytesseract.image_to_string(img, config = '--psm 6, -c tessedit_char_whitelist=0123456789%, -c load_system_dawg=0, -c load_freq_dawg=0, -c tessedit_zero_rejection=1, -c tessedit_zero_kelvin_rejection=1')

def get_date(frame):
    """Gets date from image."""
    frame = frame[:, REMOVE:]
    date = frame[DATETOP:DATEBOTTOM, DATELEFT:DATERIGHT]
    date = detect_text(date)
    # if date has dashes, return it - it means the scenario has not been played yet
    if '-' in date:
        return date
    date = date.replace('\n', '')
    # replace space with two spaces to separate the date and time
    date = date.replace(' ', '  ')
    date = date.replace('/', ' ')
    date = date.replace(':', ' ') #these characters cause problems with cv2.imwrite
    date = date.replace('р', 'P')
    date = date.replace('м', 'M') #sometimes Google Vision messes up with cyrillic characters
    date = date.replace('А', 'A')
    date = date.replace('М', 'M')
    # remove beginning and end spaces
    date = date.strip()
    return date

def get_hazard(info):
    """Gets hazard from image."""
    hazard = detect_text(info[HAZTOP:HAZBOTTOM, HAZLEFT:HAZRIGHT])
    # remove all non digit characters
    hazard = ''.join([i for i in hazard if i.isdigit()])
    return hazard

def main(filename, lang='eng'):
    # setup for google drive
    print('Authenticating with Google Drive...', end=' ')
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("creds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("creds.txt")
    drive = GoogleDrive(gauth)
    gc = pygsheets.authorize()
    # Open spreadsheet and then worksheet
    sh = gc.open('Scodes')
    wks = sh.worksheet_by_title('Codes')
    print('Done')
    cam = cv2.VideoCapture('videos/' + filename)
    # advance frames until half way through video for testing weapons
    #for i in range(0, 62):
    #    cam.read()

    ret, nextFrame = cam.read()
    if not ret:
        print('Error: No frame read from video')
        exit(0)
    if nextFrame.shape != (720, 1280, 3):
        print('Error: Video is not 720p')
        exit(0)

    #name = './tesstrain-data/scode-ground-truth/' + filename[:-4] + '_'
    nextDate = get_date(nextFrame)
    #cv2.imwrite(name + '0.tif', image_prettify(frame)) #this was for tesseract data
    equal = 0
    flagInit = True # whether there are any different frames
    flagEqual = False # indicating enough frames have been the same to move on
    flagRand = False # if there is a random rotation in the video
    currentFrame = -1
    toRemove = []
    row = 2
    # get the first empty row
    while wks.get_value('B' + str(row)) != '':
        row += 1
    initRow = row
    while(ret and not flagEqual):
        frame = nextFrame
        date = nextDate
        ret, nextFrame = cam.read()
        currentFrame += 1
        if ret:
            nextDate = get_date(nextFrame)
        # if reached end of the video, get data from current frame, then break
        if date == nextDate and ret:
            if not flagInit:
                equal += 1
            if equal < 15:
                # when it reaches 15, time for weapons phase
                continue
            else:
                flagEqual = True
                cam = cv2.VideoCapture('videos/' + filename)
                for i in range(0, toRemove[-1][1]+1): # set to frame of second to last date, +1 to get to the last date
                    cam.read()
                ret, frame = cam.read()
                currentFrame = toRemove[-1][1] + 1
        dateFile = date + '.png'
        #cv2.imwrite(name + str(currentFrame) + '.tif', image_prettify(code)) #this was for tesseract data
        if os.path.isfile(dateFile):
            print('Probably an error that ' + date + ' file already exists on frame ' + str(currentFrame))
            break

        equal = 0
        if '-' in date:
            continue
        print('Getting data for frame ' + str(currentFrame) + '...', end=' ')
        info = frame[:, REMOVE:, :]
        # get data
        # check whether file already exists locally, meaning during this run
        # indicates weapons phase has already been reached
        # determine number of waves with the checkmarks indicating whether a wave was passed
        red = np.count_nonzero(info[WAVETOP:WAVEBOTTOM, WAVELEFT:WAVERIGHT, 2] > 200)
        green = np.count_nonzero(info[WAVETOP:WAVEBOTTOM, WAVELEFT:WAVERIGHT, 1] > 200)
        if red > 100:
            red = 1
        else:
            red = 0
        green = green / 100
        green = np.round(green).astype(int)
        waves = red + green
        # map
        map = detect_text(info[MAPTOP:MAPBOTTOM, MAPLEFT:MAPRIGHT, :])
        # rotation weapons
        print('Getting rotation weapons...', end=' ')
        rot1 = info[RTOP:RBOTTOM, RLEFT:RLEFT+RWIDTH, :]
        rot2 = info[RTOP:RBOTTOM, RLEFT+RWIDTH:RLEFT+2*RWIDTH, :]
        rot3 = info[RTOP:RBOTTOM, RLEFT+2*RWIDTH:RLEFT+3*RWIDTH, :]
        rot4 = info[RTOP:RBOTTOM, RLEFT+3*RWIDTH:RLEFT+4*RWIDTH, :]
        max = [[0, ''], [0, ''], [0, ''], [0, '']]
        # compare to weapon images
        rand = False
        for rotfilename in os.listdir('./rotation-images'):
            if rotfilename.endswith('.png'):
                weap = cv2.imread('./rotation-images/' + rotfilename)
                ssim = (sk.metrics.structural_similarity(rot1, weap, channel_axis = 2), sk.metrics.structural_similarity(rot2, weap, channel_axis = 2), sk.metrics.structural_similarity(rot3, weap, channel_axis = 2), sk.metrics.structural_similarity(rot4, weap, channel_axis = 2))
                for i in range(0, 4):
                    if ssim[i] > max[i][0]:
                        max[i][0] = ssim[i]
                        max[i][1] = rotfilename[:-4]
        print('Done')
        if max[0][1] == 'Random': # assumes that if one is random, all are random, update if this changes
            flagRand = True
            rand = True
            rots = 'Random'
        else:
            rots = max[0][1] + ', ' + max[1][1] + ', ' + max[2][1] + ', ' + max[3][1]
        # waves
        top = WAVETYPETOP[waves - 1]
        left = WAVETYPELEFT
        wave1 = info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH, :]
        wave1 = detect_text(info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH, :])
        wave2 = '-'
        wave3 = '-'
        wave4 = '-'
        if waves > 1:
            top = top + WAVEHEIGHT2
            wave2 = detect_text(info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH, :])
        if waves > 2:
            top = top + WAVEHEIGHT2
            wave3 = detect_text(info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH, :])
        if waves > 3:
            top = top + WAVEHEIGHT2
            wave4 = detect_text(info[top:top+WAVEHEIGHT, left:left+WAVEWIDTH, :])
        if waves < 4 and red == 1:
            wave4 = '?' # if loss occurred, boss is unknown
        # hazard level
        #haz = detect_text(info[HAZTOP:HAZBOTTOM, HAZLEFT:HAZRIGHT, :])
        #haz = haz.replace('\n', '')
        #haz = haz.replace(' ', '')
        #if haz[-1] == '%':
        #    haz = haz[:-1]
        haz = get_hazard(info)
        hazError = False
        try:
            hazn = int(haz)
            if hazn < 1 or hazn > 333:
                hazError = True
        except:
            hazError = True
        if hazError:
            print('For frame ' + str(currentFrame) + ', hazard level is invalid. Check out directory for the image.')
            cv2.imwrite('haz' + str(currentFrame) + '_' + filename[:-4] + '.png', info[HAZTOP:HAZBOTTOM, HAZLEFT:HAZRIGHT, :])
        version = detect_text(info[VTOP:VBOTTOM, VLEFT:VRIGHT, :])
        code = info[CODETOP:CODEBOTTOM, CODELEFT:CODERIGHT, :]
        cv2.imwrite(dateFile, code)
        toRemove.append([dateFile, currentFrame, rand, waves, row])
        # save image of code to google drive
        file1 = drive.CreateFile({'title' : dateFile, 'parents': [{'id': FOLDERID}]})
        file1.SetContentFile(dateFile)
        file1['mimeType'] = 'image/png'
        file1.Upload()
        print('Uploaded code image to Google Drive')

        wks.update_value('A' + str(row), '=IMAGE("https://drive.google.com/uc?export=view&id=' + file1['id'] + '")')
        wks.update_value('B' + str(row), date)
        wks.update_value('C' + str(row), map)
        wks.update_value('D' + str(row), rots)
        if not rand:
            wks.update_value('E' + str(row), '-')
            wks.update_value('F' + str(row), '-')
            wks.update_value('G' + str(row), '-')
            wks.update_value('H' + str(row), '-')
        else:
            if waves < 4:
                wks.update_value('H' + str(row), '-')
            if waves < 3:
                wks.update_value('G' + str(row), wave3)
            if waves < 2:
                wks.update_value('F' + str(row), wave2)
        # next loop takes care of random weapons
        wks.update_value('I' + str(row), wave1)
        wks.update_value('J' + str(row), wave2)
        wks.update_value('K' + str(row), wave3)
        wks.update_value('L' + str(row), wave4)
        wks.update_value('M' + str(row), haz)
        wks.update_value('N' + str(row), version)
        row += 1
        flagInit = False
        print('Done with ' + date + ' or frame ' + str(currentFrame))

    wks.adjust_row_height(initRow, end=row-1, pixel_size=30)
    if flagInit:
        print('Error: Images are all the same scenario code')
        exit(0)

    # unnecessary to run the rest of the code if there are no random weapons
    if not flagRand:
        for i in range(0, len(toRemove)):
            os.remove(toRemove[i][0])
        return

    cam = cv2.VideoCapture('videos/' + filename)
    for i in range(0, toRemove[-1][1]+1): # set frame to first frame of last scenario before weapons phase
        cam.read()
    ret, nextFrame = cam.read()
    nextDate = get_date(nextFrame)
    currentFrame = toRemove[-1][1]
    flagInit = False
    flagEqual = False
    equal = 0
    for dateOrigFile, prevFrame, rand, waves, row in reversed(toRemove):
        if flagEqual:
            break
        dateOrig = dateOrigFile[:-4]
        print('Checking for random weapons in ' + dateOrig + ' or frame ' + str(currentFrame))
        frame = nextFrame
        date = nextDate
        while '-' in date:
            ret, nextFrame = cam.read()
            nextDate = get_date(nextFrame)
            if date != nextDate:
                date = nextDate
            currentFrame += 1
        if dateOrig != date:
            print('Error: Frame for wave types is not the same as frame for weapons. Check out directory for more info.')
            print('dateOrig:' + dateOrig + ', date:' + date)
            with open('out/log.txt', 'w') as f:
                f.write('date: ' + date + ', frame: ' + str(currentFrame) + ', prevFrame: ' + str(prevFrame) + ', row: ' + str(row))
            cv2.imwrite('out/frame' + str(currentFrame) + filename[:-4] + '.png', frame)
            cam = cv2.VideoCapture('videos/' + filename)
            for i in range(0, prevFrame):
                cam.read()
            ret, prevF = cam.read()
            cv2.imwrite('out/prevFrame' + str(prevFrame) + filename[:-4] + '.png', prevF)
            break
        ret, nextFrame = cam.read()
        currentFrame += 1
        if ret:
            nextDate = get_date(nextFrame)
        # if reached end of the video, get data from current frame, then break
        while date == nextDate and ret:
            frame = nextFrame
            # check if the frame for wave types is the same as the frame for weapons
            if equal >= 15 and flagInit: # if 15 frames are the same (but not the first 15 frames of the weapons phase) then get the data from current frame and break
                flagEqual = True
                break
            ret, nextFrame = cam.read()
            if not ret:
                break
            nextDate = get_date(nextFrame)
            currentFrame += 1
            equal += 1
        equal = 0
        flagInit = True
        if not rand:
            continue

        print('Getting random weapons for ' + date + '...', end=' ')
        info = frame[:, REMOVE:, :]
        right = WRIGHT
        top = WTOP
        weaps = []
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
            for filen in os.listdir('./weapon-images'):
                if filen.endswith('.png'):
                    weap = cv2.imread('./weapon-images/' + filen)
                    # print shape of both
                    ssim = (sk.metrics.structural_similarity(weap1, weap, channel_axis = 2), sk.metrics.structural_similarity(weap2, weap, channel_axis = 2), sk.metrics.structural_similarity(weap3, weap, channel_axis = 2), sk.metrics.structural_similarity(weap4, weap, channel_axis = 2))
                    for j in range(0, 4):
                        if ssim[j] > max[j][0]:
                            max[j][0] = ssim[j]
                            max[j][1] = filen[:-4]
            weaps.append(max[0][1] + ', ' + max[1][1] + ', ' + max[2][1] + ', ' + max[3][1])

        wks.update_value('E' + str(row), weaps[-1])
        if waves > 1:
            wks.update_value('F' + str(row), weaps[-2])
        if waves > 2:
            wks.update_value('G' + str(row), weaps[-3])
        if waves > 3:
            wks.update_value('H' + str(row), weaps[-4])
        print('Done')
    for i in range(0, len(toRemove)):
        os.remove(toRemove[i][0])

if __name__ == '__main__':
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print('Usage: python3 vid2data.py <video file> [lang=eng]')
        exit(0)
    lang = 'eng'
    if len(sys.argv) == 3:
        lang = sys.argv[2]
    main(sys.argv[1], lang)