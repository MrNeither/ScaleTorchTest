import os
import glob
import soundfile as sf  # pip install pysoundfile


def ogg2wav(ogg_path: str, wav_path: str):
    data, samplerate = sf.read(ogg_path)
    sf.write(wav_path, data, samplerate)


if __name__ == "__main__":
    root_folder = os.path.abspath('audioDB/test')
    for file in glob.glob(os.path.join(root_folder, '*.ogg')):
        source_file = os.path.join(root_folder, file)
        ogg2wav(source_file, os.path.join(root_folder, file.split('.')[0] + '.wav'))
        os.remove(source_file)
