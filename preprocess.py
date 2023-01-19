import cv2
import numpy as np
import os
import vid2data as v
from google.cloud import vision
import time



# in ./tesstrain-data/scode-ground-truth, create plain text files with the same name as the image files
# and copy the text from the image into the text file

#for filename in os.listdir('tesstrain-data/scode-ground-truth'):
#    if filename.endswith('.tif'):
#        img = cv2.imread('tesstrain-data/scode-ground-truth/' + filename)
#        #code = pytesseract.image_to_string(img, config = '--psm 6, -c tessedit_char_whitelist=-0123456789ABCDEFGHJKLMNPQRSTUVWXY, -c load_system_dawg=0, -c load_freq_dawg=0, -c load_unambig_dawg=0, -c load_punc_dawg=0, -c load_number_dawg=0, -c load_bigram_dawg=0, -c tessedit_unrej_any_wd=1, -c tessedit_enable_bigram_correction=0, -c tessedit_enable_dict_correction=0, -c debug_acceptable_wds=1')
#        code = pytesseract.image_to_string(img, config = '--psm 6, -c tessedit_char_whitelist=-0123456789ABCDEFGHJKLMNPQRSTUVWXY, -c load_system_dawg=0, -c load_freq_dawg=0')
#        cv2.imwrite('tesstrain-data/scode-ground-truth/' + code + '.tif', img)
#        # remove .tif file
#        os.remove('tesstrain-data/scode-ground-truth/' + filename)


# make ground truth files

#make_gt_files('tesstrain/data/scode-ground-truth')
#img = cv2.imread('tesstrain/data/scode-ground-truth/SK7Y-UF1R-HES5-1Q5N.tif')
##img = image_prettify(img)
#print(pytesseract.image_to_string('tesstrain/data/scode-ground-truth/SK7Y-UF1R-HES5-1Q5N.tif', lang = 'scode'))

def detect_text(content, full=False):
    """Detects text in the image."""
    # convert content to bytes with cv2
    content = cv2.imencode('.png', content)[1].tobytes()
    #with io.open(path, 'rb') as image_file:
    #    content = image_file.read()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=v.CREDS
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    time.sleep(0.01)
    image_context = {
        "text_detection_params": {
            "enable_text_detection_confidence_score": True
        },
        "language_hints": ["en"]
    }
    response = client.text_detection(image=image, image_context=image_context) #type: ignore
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    print(response.full_text_annotation.pages[0])
    if full:
        response.full_text_annotation.pages[0].blocks[0].paragraphs[0]
    else:
        return response.text_annotations[0].description


def detect_code(content):
    paragraph = detect_text(content, full=True)
    if paragraph is None:
        raise Exception('No text detected')
    if len(paragraph.words) != 7:
        raise Exception('Incorrect number of words for code')
    if len(paragraph.words[0].symbols) != 4 or len(paragraph.words[2].symbols) != 4 or len(paragraph.words[4].symbols) != 4 or len(paragraph.words[6].symbols) != 4:
        raise Exception('Incorrect number of symbols for code')
    try:
        print(paragraph.words[0].symbols[0])
        print(len(paragraph.words[1].symbols))
        # print response language
        print('locale: ', response.text_annotations[0].locale)
    except:
        print('exception')
    texts = response.text_annotations
    return texts[0].description

if __name__ == '__main__':
    img = cv2.imread('testing-images/date.png')
    #cam = cv2.VideoCapture('videos/codes5.mp4')
    #ret, img = cam.read()
    #for i in range(0, 200):
    #    ret, img = cam.read()

    #remove = v.REMOVE
    #img = img[:, remove:]
    # get weapons
    waves = 3
    #cam = cv2.VideoCapture('videos/output2.mp4')
    #ret, img = cam.read()
    text = detect_text(img)
    print(text)
    title = 'test'
    os.system("""
              osascript -e 'display notification "{}" with title "{}" sound name "Glass"'
              """.format(text, title))
    time.sleep(2)








    """Add weapons to database"""
    #right = v.WRIGHT
    #top = v.WTOP
    #for i in range(0, waves):
    #    weap1 = img[top:top+v.WHEIGHT, right-v.WWIDTH:right]
    #    top = top + v.WHEIGHT + v.WHEIGHT2
    #    weap2 = img[top:top+v.WHEIGHT, right-v.WWIDTH:right]
    #    top = top + v.WHEIGHT + v.WHEIGHT2
    #    weap3 = img[top:top+v.WHEIGHT, right-v.WWIDTH:right]
    #    top = top + v.WHEIGHT + v.WHEIGHT2
    #    weap4 = img[top:top+v.WHEIGHT, right-v.WWIDTH:right]

    #    cv2.imwrite('weapon-images/' + str(i) + 'weap1.png', weap1)
    #    cv2.imwrite('weapon-images/' + str(i) + 'weap2.png', weap2)
    #    cv2.imwrite('weapon-images/' + str(i) + 'weap3.png', weap3)
    #    cv2.imwrite('weapon-images/' + str(i) + 'weap4.png', weap4)
    #    right = right - v.WWIDTH
    #    top = v.WTOP

    # save images to weapon-images
