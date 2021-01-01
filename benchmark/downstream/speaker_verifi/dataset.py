# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speaeker_verifi dataset ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
import sys
#-------------#
import numpy as np
import pandas as pd
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import os
import torch
import random
import torchaudio
from tqdm import trange
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from librosa.util import find_files
from functools import lru_cache 
import audiosegment
import copy
import IPython
import pdb
import os
import glob
import pkg_resources
import six
import time
from multiprocessing import Pool
import soundfile as sf
from torch.utils.data.sampler import Sampler, BatchSampler
from pathlib import Path
from sox import Transformer
import concurrent.futures
# torchaudio.set_audio_backend("sox_io")
max_timestep = int(16000 * 8)
# Voxceleb 1 + 2 
# preprocessing need seperate folder to dev, train, test
class AudioBatchData(Dataset):
    def __init__(self, file_path, max_timestep=16000*5, meta_data=None, utter_number=5, sizeWindow=1,nProcessLoader=10, MAX_SIZE_LOADED=20, batch_size=16, repeat=5):

        self.roots = file_path
        self.root_key = list(self.roots.keys())
        
        # extract dev speaker and store in self.black_list_spealers
        with open(meta_data, "r") as f:
            self.black_list_speakers = f.read().splitlines()

        # calculate speakers and support to remove black list speaker
        self.all_speakers = \
            [f.path for key in self.root_key for f in os.scandir(self.roots[key]) if f.is_dir() and f.path.split("/")[-1] not in self.black_list_speakers]
        
        self.utter_number = utter_number
        self.necessary_dict = self.processing()
        self.dataset = self.necessary_dict['spk_paths']
        self.batch_size = batch_size
        self.sizeWindow = sizeWindow
        self.MAX_SIZE_LOADED= MAX_SIZE_LOADED
        self.repeat = repeat

        start = time.time()
        self.file_list = {}
        for x in tqdm(self.dataset):
            self.file_list[x] = find_files(x, ext='wav')
        end = time.time()

        print(f"search all audio file need {end-start} seconds")
        self.max_timestep = max_timestep

        # prepare n processor for load chunk data from disk
        self.nProcessLoader = nProcessLoader
        # self.reload_pool = concurrent.futures.ThreadPoolExecutor(nProcessLoader)
        self.packageIndex, self.totSize = [], 0
        self.batched_paths = []
        start = time.time()
        self.prepare()
        self.loadNextPack(first=True)

        self.loadNextPack()
        end = time.time()
        print(f"preload data chunk takes {end-start} seconds")
    
    def prepare(self):

        random.shuffle(self.dataset)        
        start_time = time.time()

        start, packageSize = 0, 0
        for index, speaker_id in tqdm(enumerate(self.all_speakers)):
            # take negative ids
            all_ids = random.sample(self.all_speakers[:index] + self.all_speakers[index+1:], self.batch_size-1)
            
            # take positive id
            positive_id = speaker_id
            all_ids.append(positive_id)
            all_paths = []
            for idx in all_ids:
                paths=random.sample(self.file_list[idx], self.utter_number)
                all_paths.extend(paths)
                # all_paths = list(map(Path,all_paths))

            self.batched_paths.append(all_paths)
            packageSize += 1
            if packageSize > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, index])
                self.totSize += packageSize
                start, packageSize = index, 0
        
        if packageSize > 0:
            self.packageIndex.append([start, len(self.all_speakers)])
            self.totSize += packageSize

        print(f'Scanned {self.totSize} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f"{len(self.packageIndex)} chunks computed")
        self.currentPack = -1
        self.nextPack = 0
    
    def take_path(self, idx):
        paths=random.sample(self.file_list[idx], self.utter_number)
        return paths
        
    def processing(self):
        
        speaker_num = len(self.all_speakers)
        return {"spk_paths":self.all_speakers,"total_spk_num":speaker_num,"pair_table":None}
    
    def clear(self):
        if 'data' in self.__dict__:
            del self.data
        if 'length' in self.__dict__:
            del self.length
    
    def loadNextPack(self, first=False):
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            start_time = time.time()
            print('Joining pool')
            # self.r.wait()                
            self.nextData = self.r
            self.parseNextDataBlock()
            print(f'Joined process, elapsed={time.time()-start_time:.3f} secs')
            del self.nextData
        self.nextPack = (self.currentPack + 1) % len(self.packageIndex)
        seqStart, seqEnd = self.packageIndex[self.nextPack]
        if self.nextPack == 0 and len(self.packageIndex) > 1:
            self.prepare()
        # print(seqStart, seqEnd)
        # IPython.embed()
        # pdb.set_trace()
        with concurrent.futures.ThreadPoolExecutor(self.nProcessLoader) as multithread:
            self.r = multithread.map(loadFile_thread_exec,self.batched_paths[seqStart:seqEnd])
                            
    
    def parseNextDataBlock(self):

        # To accelerate the process a bit
        tmpData = []
        tmpLength = []

        for batch_seq, batch_length in self.nextData:
            # batch_seq = self.nextData[0][index]
            # batch_length = self.nextData[1][index]
            tmpData.append(batch_seq)
            tmpLength.append(batch_length)
            
            del batch_seq
            del batch_length
        
        sample_num=len(tmpData)

        for _ in range(self.repeat):
            for index in range(sample_num):
                batch_seq = tmpData[index]
                batch_length = tmpLength[index]
                tmpData.append(batch_seq)
                tmpLength.append(batch_length)
                tmpData.append(batch_seq)
                tmpLength.append(batch_length)

            # if repeat == sample_num-2:
            #     del batch_seq
            #     del batch_length

        self.data = tmpData
        self.length = tmpLength

    
    def __len__(self):
        return self.necessary_dict['total_spk_num']


    def __getitem__(self, index):

        if index < 0 or index >= len(self.data) - self.sizeWindow - 1:
            # print(index)
            pass

        x_list = self.data[index:(self.sizeWindow+index)][0]
        length_list = self.length[index:(self.sizeWindow+index)][0]

        # print(x_list)
        # print(length_list)

        return x_list, length_list
    
    def getDataLoader(self, batchSize, type, numWorkers=0,
                      onLoop=-1):
        r"""
        Get a batch sampler for the current dataset.
        Args:
            - batchSize (int): batch size
            - groupSize (int): in the case of type in ["speaker", "sequence"]
            number of items sharing a same label in the group
            (see AudioBatchSampler)
            - type (string):
                type == "speaker": grouped sampler speaker-wise
                type == "sequence": grouped sampler sequence-wise
                type == "sequential": sequential sampling
                else: uniform random sampling of the full audio
                vector
            - randomOffset (bool): if True add a random offset to the sampler
                                   at the begining of each iteration
        """
        nLoops = len(self.packageIndex)
        totSize = self.totSize // (self.sizeWindow * batchSize)
        if onLoop >= 0:
            self.currentPack = onLoop - 1
            self.loadNextPack()
            nLoops = 1

        def samplerCall():
            offset = 0
            return self.getBaseSampler(type, batchSize, offset)

        return AudioLoader(self, samplerCall, nLoops, self.loadNextPack,
                           totSize, numWorkers)
    
    def getBaseSampler(self, type, batchSize, offset):

        sampler = UniformAudioSampler(len(self.data), self.sizeWindow,
                                      offset)
        return BatchSampler(sampler, batchSize, True)

def collate_fn(data_sample):

    wavs = []
    lengths = []

    for samples in data_sample:
        wavs.extend(samples[0])
        lengths.extend(samples[1])

    return wavs, lengths, -1,
    
def loadFile(data_list):

    wav_list=[]
    length_list=[]
    # print(len(data_list))
    for i in trange(len(data_list)):
        wavs = []
        lengths = []
        for j in range(len(data_list[i])):
            fullPath = data_list[i][j]
            # print("data is ", data[i][j])
            # print(fullPath)
            transformer = Transformer()
            transformer.norm()
            transformer.silence(silence_threshold=1, min_silence_duration=0.1)
            transformer.set_output_format(rate=16000, bits=16, channels=1)
            wav = transformer.build_array(input_filepath=str(fullPath))
            wav = torch.tensor(wav / (2 ** 15)).float()
            # print(fullPath)
            # wav =torch.randn(16000*8)
            # wav=torch.tensor(sf.read(fullPath)[0]).float()
            length = len(wav)
            if length > max_timestep:
                start = 0
                end = max_timestep
                length = max_timestep
                wav=wav[start:end]
            wavs.append(wav)
            lengths.append(torch.tensor(length).long())
        wav_list.append(wavs)
        length_list.append(lengths)
        

    return wav_list, length_list

def loadFile_thread_exec(data):
    
    # print(len(data_list))
    wavs = []
    lengths = []
    for i in range(len(data)):
        
        fullPath = data[i]
        # print("data is ", data[i][j])
        # print(fullPath)
        transformer = Transformer()
        transformer.norm()
        transformer.silence(silence_threshold=1, min_silence_duration=0.1)
        transformer.set_output_format(rate=16000, bits=16, channels=1)
        wav = transformer.build_array(input_filepath=str(fullPath))
        wav = torch.tensor(wav / (2 ** 15)).float()
        # print(fullPath)
        # wav =torch.randn(16000*8)
        # wav=torch.tensor(sf.read(fullPath)[0]).float()
        length = len(wav)
        if length > max_timestep:
            start = 0
            end = max_timestep
            length = max_timestep
            wav=wav[start:end]
        wavs.append(wav)
        lengths.append(torch.tensor(length).long())
        # wav_list.append(wavs)
        # length_list.append(lengths)
        

    return wavs, lengths




class UniformAudioSampler(Sampler):

    def __init__(self,
                 dataSize,
                 sizeWindow,
                 offset):

        self.len = dataSize // sizeWindow
        self.sizeWindow = sizeWindow
        self.offset = offset
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        return iter((self.offset
                     + self.sizeWindow * torch.randperm(self.len)).tolist())

    def __len__(self):
        return self.len

class AudioLoader(object):
    r"""
    A DataLoader meant to handle an AudioBatchData object.
    In order to handle big datasets AudioBatchData works with big chunks of
    audio it loads sequentially in memory: once all batches have been sampled
    on a chunk, the AudioBatchData loads the next one.
    """
    def __init__(self,
                 dataset,
                 samplerCall,
                 nLoop,
                 updateCall,
                 size,
                 numWorkers):
        r"""
        Args:
            - dataset (AudioBatchData): target dataset
            - samplerCall (function): batch-sampler to call
            - nLoop (int): number of chunks to load
            - updateCall (function): function loading the next chunk
            - size (int): total number of batches
            - numWorkers (int): see torch.utils.data.DataLoader
        """
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers

    def __len__(self):
        return self.size

    def __iter__(self):

        for i in range(self.nLoop):
            sampler = self.samplerCall()
            dataloader = DataLoader(self.dataset,
                                    batch_sampler=sampler,
                                    num_workers=self.numWorkers, \
                                    collate_fn=collate_fn)
            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()
# class SpeakerVerifi_train(Dataset):
#     def __init__(self, file_path, max_timestep=16000*5, meta_data=None, utter_number=5):

#         self.roots = file_path
#         self.root_key = list(self.roots.keys())
        
#         # extract dev speaker and store in self.black_list_spealers
#         with open(meta_data, "r") as f:
#             self.black_list_speakers = f.read().splitlines()

#         # calculate speakers and support to remove black list speaker
#         self.all_speakers = \
#             [f.path for key in self.root_key for f in os.scandir(self.roots[key]) if f.is_dir() and f.path.split("/")[-1] not in self.black_list_speakers]
        
#         self.utter_number = utter_number
#         self.necessary_dict = self.processing()
#         self.dataset = self.necessary_dict['spk_paths']
        
#         start = time.time()
#         self.file_list = {}
#         for x in tqdm(self.dataset):
#             self.file_list[x] = find_files(x, ext='wav')
#         end = time.time()

#         print(f"search all audio file need {end-start} seconds")
#         self.max_timestep = max_timestep
        
#     def processing(self):
        
#         speaker_num = len(self.all_speakers)
#         return {"spk_paths":self.all_speakers,"total_spk_num":speaker_num,"pair_table":None}
    
#     def __len__(self):
#         return self.necessary_dict['total_spk_num']


#     def __getitem__(self, idx):
#         path = random.sample(self.file_list[self.dataset[idx]], self.utter_number)

#         x_list = []
#         length_list = []

#         for i in range(len(path)):
#             wav, sr = torchaudio.load(path[i])
#             wav = wav.squeeze(0)
#             length = wav.shape[0]

#             if length > self.max_timestep:
#                 length = self.max_timestep
#                 start = random.randint(0,length - self.max_timestep)
#                 wav = wav[start:start+self.max_timestep]

#             x_list.append(wav)
#             length_list.append(torch.tensor(length).long())

#         return x_list, length_list
    
#     def collate_fn(self,data_sample):

#         wavs = []
#         lengths = []
#         indexes = []

#         for samples in data_sample:
#             wavs.extend(samples[0])
#             lengths.extend(samples[1])

#         return wavs, lengths, -1,


class SpeakerVerifi_dev(Dataset):
    def __init__(self, file_path, max_timestep, meta_data=None):

        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.max_timestep = max_timestep
        self.dataset = self.necessary_dict['pair_table'] 
        
    def processing(self):
        pair_table = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1])
            pair_2= os.path.join(self.root, list_pair[2])
            one_pair = [list_pair[0],pair_1,pair_2 ]
            pair_table.append(one_pair)
        return {"spk_paths":None,"total_spk_num":None,"pair_table":pair_table}

    def __len__(self):
        return len(self.necessary_dict['pair_table'])

    def __getitem__(self, idx):
        y_label, x1_path, x2_path = self.dataset[idx]
        wav1, _ = torchaudio.load(x1_path)
        wav2, _ = torchaudio.load(x2_path)

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        length1 = wav1.shape[0]

        if length1 > self.max_timestep:
            length1 = self.max_timestep
            start = random.randint(0,length1 - self.max_timestep)
            wav1 = wav1[start:start+self.max_timestep]

        length2 = wav2.shape[0]

        if length2 > self.max_timestep:
            length2 = self.max_timestep
            start = random.randint(0,length2 - self.max_timestep)
            wav2 = wav1[start:start+self.max_timestep]


        return wav1, wav2, \
        torch.tensor(length1), torch.tensor(length2), \
        torch.tensor(int(y_label[0])),
    
    def collate_fn(self, data_sample):
        wavs1 = []
        wavs2 = []
        lengths1 = []
        lengths2 = []
        ylabels = []

        for samples in data_sample:
            wavs1.append(samples[0])
            wavs2.append(samples[1])
            lengths1.append(samples[2])
            lengths2.append(samples[3])
            ylabels.append(samples[4])

        all_wavs = []
        all_wavs.extend(wavs1)
        all_wavs.extend(wavs2)

        all_lengths = []
        all_lengths.extend(lengths1)
        all_lengths.extend(lengths2)

        return all_wavs, all_lengths, ylabels



class SpeakerVerifi_test(Dataset):
    def __init__(self, file_path,meta_data=None):

        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.dataset = self.necessary_dict['pair_table'] 
        
    def processing(self):
        pair_table = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1])
            pair_2= os.path.join(self.root, list_pair[2])
            one_pair = [list_pair[0],pair_1,pair_2 ]
            pair_table.append(one_pair)
        return {"spk_paths":None,"total_spk_num":None,"pair_table":pair_table}

    def __len__(self):
        return len(self.necessary_dict['pair_table'])

    def __getitem__(self, idx):
        y_label, x1_path, x2_path = self.dataset[idx]
        wav1, _ = torchaudio.load(x1_path)
        wav2, _ = torchaudio.load(x2_path)

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        length1 = wav1.shape[0]
        length2 = wav2.shape[0]

        return wav1, wav2, \
        torch.tensor(length1), torch.tensor(length2), \
        torch.tensor(int(y_label[0])),
    
    def collate_fn(self, data_sample):
        wavs1 = []
        wavs2 = []
        lengths1 = []
        lengths2 = []
        ylabels = []

        for samples in data_sample:
            wavs1.append(samples[0])
            wavs2.append(samples[1])
            lengths1.append(samples[2])
            lengths2.append(samples[3])
            ylabels.append(samples[4])

        all_wavs = []
        all_wavs.extend(wavs1)
        all_wavs.extend(wavs2)

        all_lengths = []
        all_lengths.extend(lengths1)
        all_lengths.extend(lengths2)

        return all_wavs, all_lengths, ylabels

