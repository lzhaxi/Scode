from pydrive.auth import GoogleAuth
from pydrive.auth import RefreshError
from pydrive.drive import GoogleDrive
import os

if __name__ == '__main__':
    # setup for google drive
    print('Authenticating with Google Drive...', end=' ')
    gauth = GoogleAuth(settings_file='settings.yaml')
    gauth.LoadCredentialsFile("creds.json")
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
    gauth.SaveCredentialsFile("creds.json")
    drive = GoogleDrive(gauth)
    print('Done')
    folder_id = '1pDpVFc6g0ZN1RojRfZF10t4KizcCY-fnWYUpDIbG-bh3SmBvL6kWxYZOMbysh1omwU36j5va'
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    if len(file_list) == 0:
        print('No video files to download.')
    else:
        for file in file_list:
            file_name = file['title']
            print('Downloading: ' + file_name)
            file_path = os.path.join('videos', file_name)
            file.GetContentFile(file_path)
            file.Trash()
        print('Done downloading videos')

    folder_id = '1U0LEojJ5emonut_aRujYRv-DAkzA0qRs_-4UGgv2vWdbqDVMh0k7F31-OazEIOjGLSb4xVWt'

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    if len(file_list) == 0:
        print('No image files to download.')
    else:
        for file in file_list:
            file_name = file['title']
            print('Downloading: ' + file_name)
            file_path = os.path.join('pictures', file_name)
            file.GetContentFile(file_path)
            file.Trash()
        print('Done downloading')
