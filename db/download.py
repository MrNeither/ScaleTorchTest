import os
import requests
from db_paths import *


def download_from_url(url: str, save_path: str) -> bool:
    # Todo: param makedirs if not exist path to save_path

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        print(f'download from {url} to {save_path}')
        with open(save_path, 'wb') as f:
            f.write(response.raw.read())
        return True
    print(f"Response url: {url} return bad status = {response.status_code}")
    return False


def download_emov(save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    for file in EmoV_DB_FILES:
        source_path = os.path.join(EmoV_DB_ROOT, file)
        target_path = os.path.join(save_path, file)
        download_from_url(source_path, target_path)


def download_ravdness(save_path: str, only_audio=True) -> None:
    if only_audio:
        files = RAVDNESS_FILES[:2]
    else:
        files = RAVDNESS_FILES
    os.makedirs(save_path, exist_ok=True)
    for file in files:
        source_path = os.path.join(RAVDNESS_ROOT, file)
        target_path = os.path.join(save_path, file.split('?')[0])
        download_from_url(source_path, target_path)


if __name__ == '__main__':
    # Todo: get params path
    db_path = 'audioDB'
    # download_emov(os.path.join(db_path, 'emovdb')
    download_ravdness(os.path.join(db_path, 'ravdness'))
