EmoV_DB_ROOT = 'https://www.openslr.org/resources/115/'
EmoV_DB_FILES = [
    'bea_Amused.tar.gz',
    'bea_Angry.tar.gz',
    'bea_Disgusted.tar.gz',
    'bea_Neutral.tar.gz',
    'bea_Sleepy.tar.gz',
    'jenie_Amused.tar.gz',
    'jenie_Angry.tar.gz',
    'jenie_Disgusted.tar.gz',
    'jenie_Neutral.tar.gz',
    'jenie_Sleepy.tar.gz',
    'josh_Amused.tar.gz',
    'josh_Neutral.tar.gz',
    'josh_Sleepy.tar.gz',
    'sam_Amused.tar.gz',
    'sam_Angry.tar.gz',
    'sam_Disgusted.tar.gz',
    'sam_Neutral.tar.gz',
    'sam_Sleepy.tar.gz',
]

RAVDNESS_ROOT = "https://zenodo.org/record/1188976/files/"
RAVDNESS_FILES = (
    'Audio_Song_Actors_01-24.zip?download=1',
    'Audio_Speech_Actors_01-24.zip?download=1',
    *(f'Video_Song_Actor_{str(i).zfill(2)}.zip?download=1' for i in range(1, 25)),
    *(f'Video_Speech_Actor_{str(i).zfill(2)}.zip?download=1' for i in range(1, 25)),
)
