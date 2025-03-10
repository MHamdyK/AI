import kagglehub

path = kagglehub.dataset_download('chrisfilo/urbansound8k')
print(f"Path do dataset files:{path}")

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class UrbanSoundDataset(Dataset):

  # annotation_file is the .csv file, audio_dir is the directory that has the .WAV files
  def __init__(self,annotation_file,audio_dir):
    self.annotations = pd.read_csv(annotation_file)
    self.audio_dir = audio_dir

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):

    audio_sample_path = self._get_audio_sample_signal_path(index)
    label = self._get_audio_sample_label(index)
    signal,sr = torchaudio.load(audio_sample_path)

    return signal,label

  def _get_audio_sample_signal_path(self,index):
    fold_path = f"fold{self.annotations.iloc[index,5]}"
    audio_path = self.annotations.iloc[index,0]
    full_audio_path = os.path.join(self.audio_dir,fold_path,audio_path) # ignore:type
    return full_audio_path

  def _get_audio_sample_label(self,index):
    label = self.annotations.iloc[index,6]
    return label

if __name__ == "__main__":
  path_to_csv = os.path.join(path,"UrbanSound8K.csv")
  # /root/.cache/kagglehub/datasets/chrisfilo/urbansound8k/versions/1/UrbanSound8K.csv
  print(f"Path to csv file:{path_to_csv}")
  usd = UrbanSoundDataset(path_to_csv,path)
  print(len(usd))
  signal,label = usd[1]
  print(f"label:{label}")

