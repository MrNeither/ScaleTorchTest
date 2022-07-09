import os
from db.download import download_emov, download_ravdness
from db.preparing import prepare_emov, prepare_ravdness
from db.preprocessing import first_preproc

if __name__ == '__main__':
    db_path: str = os.path.abspath('db')
    raw_emov = os.path.join(db_path, 'raw_emov')
    raw_ravdness = os.path.join(db_path, 'raw_ravdness')
    prepared_path:str = os.path.join(db_path, 'prepared')

    #download_emov(raw_emov)
    #download_ravdness(raw_ravdness)
    #prepare_emov(raw_emov, prepared_path)
    #prepare_ravdness(raw_ravdness, prepared_path)

    v1_dataset_path = os.path.join(db_path, 'v1_dataset')
    first_preproc(prepared_path, v1_dataset_path)
