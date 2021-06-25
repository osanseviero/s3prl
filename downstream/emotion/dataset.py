# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
import pandas as pd
from pathlib import Path
from os.path import join as path_join

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import Resample, Vol

SAMPLE_RATE = 16000


class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True, normalize=False):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        self.normalize = normalize
        self.vol = Vol(-30, "db")
        if self.pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        
        if self.normalize:
            wav = F.layer_norm(wav, wav.shape)
            wav = self.vol(wav.unsqueeze(0)).squeeze(0)

        return wav.numpy(), label

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    wavs, labels = [], []
    for wav, label in samples:
        wavs.append(wav)
        labels.append(label)
    return wavs, labels


class EmotionHidden(Dataset):
    def __init__(self, hidden_path, dev=True, normalize=False):
        self.hidden_path = Path(hidden_path)
        self.table = pd.read_csv(self.hidden_path / ("emotion_dev.csv" if dev else "emotion_test.csv"))
        self.sample_rate = 44100
        self.resample = Resample(self.sample_rate, SAMPLE_RATE)
        self.normalize = normalize
        if normalize:
            self.vol = Vol(-30, "db")

        unused_emtions = set()
        def convert_label(label):
            if label == "neutral":
                return 0
            elif label == "joy":
                return 1
            elif label == "anger":
                return 2
            elif label == "sadness":
                return 3
            else:
                unused_emtions.add(label)
                return -1

        self.table["label"] = self.table["emotion"].apply(convert_label)
        self.table = self.table[self.table["label"] != -1]
        print(f"[Emotion] - Unused emotion: {unused_emtions}")

        self.utterance_ids = self.table["utterance_id"].tolist()
        self.labels = self.table["label"].tolist()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        wav_path = self.hidden_path / f"{self.utterance_ids[index]}.wav"
        try:
            wav, sr = torchaudio.load(wav_path)
            assert sr == self.sample_rate
        except RuntimeError:
            prefix = "".join(wav_path.split(".")[:-1])
            extention = wav_path.split(".")[-1]
            file1 = prefix + "_(1)." + extention
            file2 = prefix + "_(2)." + extention
            wav1, sr1 = torchaudio.load(file1)
            wav2, sr2 = torchaudio.load(file2)
            assert sr1 == self.sample_rate
            assert sr2 == self.sample_rate
            wav = torch.cat((wav1, wav2), dim=-1)
        wav = wav.mean(dim=0, keepdim=True)
        wav = self.resample(wav)

        if self.normalize:
            wav = F.layer_norm(wav, wav.shape)
            wav = self.vol(wav)

        wav = wav.view(-1).numpy()
        label = self.labels[index]

        return wav, label
