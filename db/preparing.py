import os
import glob
import shutil
import tarfile
import zipfile
from enum import Enum, unique


@unique
class Emotion(Enum):
    HAPPY = 0
    NEUTRAL = 1
    ANGRY = 2
    NAN = 3


def extract(path: str, to_directory: str = '.') -> None:
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else:
        raise ValueError("Could not extract `%s` as no appropriate extractor is found" % path)

    cwd = os.getcwd()
    os.chdir(to_directory)

    # todo: create message if catch error
    try:
        file = opener(path, mode)
        try:
            file.extractall()
        finally:
            file.close()
    finally:
        os.chdir(cwd)


def prepare_emov(raw_path: str, prepared_path: str) -> None:
    emotions_map = {
        'amused': Emotion.HAPPY,
        'anger': Emotion.ANGRY,
        'neutral': Emotion.NEUTRAL,
        'disgust': Emotion.NAN,
        'sleep': Emotion.NAN,
    }
    os.makedirs(prepared_path, exist_ok=True)
    for tar_file in os.listdir(raw_path):
        source_folder = os.path.join(raw_path, tar_file)
        target_folder = os.path.join(prepared_path, tar_file.split('.')[0])
        if not os.path.exists(target_folder):
            # todo: this works only for one folder layer creating
            os.mkdir(target_folder)
        print(source_folder, target_folder)
        extract(source_folder, target_folder)
        for file in os.listdir(target_folder):
            emotion = [val for key, val in emotions_map.items() if key in file and val is not Emotion.NAN]
            if len(emotion) != 1:
                os.remove(os.path.join(target_folder, file))
                continue

            os.replace(os.path.join(target_folder, file),
                       os.path.join(prepared_path, f"{emotion[0].value}_{hash(file)}.wav"))
        os.rmdir(target_folder)


def prepare_ravdness(raw_path: str, prepared_path: str) -> None:
    """
    (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised

    Filename example: 02-01-06-01-02-01-12.mp4

    Video-only (02)
    Speech (01)
    Fearful (06)
    Normal intensity (01)
    Statement "dogs" (02)
    1st Repetition (01)
    12th Actor (12)
    Female, as the actor ID number is even.
    """
    emotions_map = {
        '01': Emotion.NEUTRAL,
        '02': Emotion.NEUTRAL,
        '03': Emotion.HAPPY,
        '04': Emotion.NAN,
        '05': Emotion.ANGRY,
        '06': Emotion.NAN,
        '07': Emotion.NAN,
        '08': Emotion.NAN,
    }

    for tar_file in os.listdir(raw_path):
        source_folder = os.path.join(raw_path, tar_file)
        target_folder = os.path.join(prepared_path, tar_file.split('.')[0])
        if not os.path.exists(target_folder):
            # todo: this works only for one folder layer creating
            os.mkdir(target_folder)
        extract(source_folder, target_folder)
        for file in glob.glob(os.path.join(target_folder, '*', '*.wav')):
            meta = file.split('-')
            emotion = emotions_map[meta[2]]
            if emotion == Emotion.NAN:
                os.remove(os.path.join(target_folder, file))
                continue

            os.replace(os.path.join(target_folder, file),
                       os.path.join(prepared_path, f"{emotion.value}_{hash(file)}.wav"))
        shutil.rmtree(target_folder)


if __name__ == "__main__":
    db_path: str = os.path.abspath('audioDB')
    prepared_path: str = os.path.join(db_path, 'prepared')
    os.makedirs(prepared_path, exist_ok=True)

    emov_path: str = os.path.join(db_path, 'emovdb')
    # prepare_emov(emov_path, prepared_path)

    ravdness_path: str = os.path.join(db_path, 'ravdness')
    prepare_ravdness(ravdness_path, prepared_path)
