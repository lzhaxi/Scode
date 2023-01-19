# go through all the files in google drive, get the text from each image, and flag the ones that are similar to each other

from google.cloud import vision
import pygsheets
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import vid2data as v

def compare_strings(a, b):
    # compare two strings to see if they have the same letters for 15 or more characters
    # return true if they do, false otherwise
    if len(a) < 15 or len(b) < 15:
        return False
    count = 0
    for i in range(0, len(a)):
        if a[i] == b[i]:
            count += 1
        if count >= 15:
            return True
    return False

def detect_text(path):
    """Detects text in the image."""
    import io
    # convert content to bytes with cv2
    #content = cv2.imencode('.png', content)[1].tobytes()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=v.CREDS
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image) #type: ignore
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    texts = response.text_annotations
    return texts[0].description

if __name__ == '__main__':
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
    # get all the files in the folder
    file_list = drive.ListFile({'q': "'" + v.FOLDERID + "' in parents and trashed=false"}).GetList()
    texts = []
    # get the text from each image
    for file in file_list:
        file.GetContentFile('temp.png', mimetype='image/png')
        texts.append(detect_text('temp.png'))
        os.remove('temp.png')
    # compare each text to every other text, flagging those that have the same letters for 15 or more characters
    flag = []
    for i in range(0, len(texts)):
        for j in range(i + 1, len(texts)):
            if compare_strings(texts[i], texts[j]):
                flag.append([file_list[i]['title'], file_list[j]['title']])
    # write the flagged files to 'check.txt'
    with open('check.txt', 'w') as f:
        for pair in flag:
            f.write(pair[0] + ' ' + pair[1] + '\n')
