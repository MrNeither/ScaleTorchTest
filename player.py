import os
import numpy as np
import torch
import moviepy.editor as mp
import librosa
from db.preparing import Emotion
import cv2
from bisect import bisect_left

input_size = 137
lr_shift = int((input_size - 1) / 2)
softmax = torch.nn.Softmax(dim=1)


def imageP(path: str):
    im = cv2.imread(path, -1)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    im = cv2.resize(im, (128, 128))
    return im


imagesEmotions = {
    Emotion.HAPPY: imageP('audioDB/images/happy.png'),
    Emotion.NEUTRAL: imageP('audioDB/images/neutral.png'),
    Emotion.ANGRY: imageP('audioDB/images/angry.png')
}


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv1d(22, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        self.c3 = torch.nn.Sequential(
            torch.nn.Conv1d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        self.f1 = torch.nn.Linear(16 * 4, 32)
        self.relu = torch.nn.ReLU()
        self.f2 = torch.nn.Linear(32, 3)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = x.flatten(start_dim=1)
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        return x


def run_model_on(model, x: np.ndarray, stride=1):
    predictions = []
    padded = np.zeros((x.shape[0], x.shape[1] + input_size - 1))
    padded[:, lr_shift:x.shape[1] + lr_shift] = x

    for s in range(0, len(x[0]), stride):
        x_ = padded[:, s:s + input_size]
        x_ = torch.tensor(x_)
        x_.unsqueeze_(0)
        y_ = model(x_)
        y__ = softmax(y_).argmax(dim=1)
        predictions.append((s, y__))

    return predictions


if __name__ == '__main__':
    exp_name = 'friends'
    target_folder = os.path.abspath(os.path.join('audioDB', 'test', 'video'))
    my_clip = mp.VideoFileClip(os.path.join(target_folder, f'{exp_name}.mp4'))
    moviesize = my_clip.size

    audio_file_path = os.path.join(target_folder, f'{exp_name}.wav')
    my_clip.audio.write_audiofile(audio_file_path)

    y, sr = librosa.load(audio_file_path, sr=44100)
    x = np.concatenate([
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr)
    ])

    model = torch.load('models/exp1.model')
    model.eval()

    predictions = run_model_on(model, x)

    frames = [x[0] for x in predictions]
    frames_time = librosa.frames_to_time(frames)
    clip = mp.VideoClip(
        lambda t: imagesEmotions[Emotion(predictions[bisect_left(frames_time, t)][1].item())])
    clip.set_pos(lambda t: (
        moviesize[0], moviesize[1]))

    final = mp.CompositeVideoClip([my_clip, clip])
    final.subclip(0, my_clip.duration).write_videofile(os.path.join(target_folder, f"{exp_name}_emojisible.mov"),
                                                       fps=24,
                                                       codec='libx264')
