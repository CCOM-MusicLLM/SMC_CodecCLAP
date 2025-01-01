#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json, os, shutil
import torch
import random
import librosa
import os
# os.environ['SNDFILE_ERROR_LEVEL'] = '0'  # 设置为最低错误级别
import soundfile
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning, module="soundfile")
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler, BatchSampler
from data_handling.sampler import BySequenceLengthSampler, BySequenceBatchSampler
from data_handling.text_transform import text_preprocess
import torch.nn.functional as F
from audiocraft.data.audio import audio_read, _av_read
from audiocraft.data.audio_utils import convert_audio


def _load_json_file(files, blacklist=None):
    json_data = []
    audio_id = 0
    blacklist = None
    if blacklist is not None:
        with open(blacklist, 'r') as f:
            blacklist = json.load(f)
    for file in files:
        with open(file, "r") as f:
            json_obj = json.load(f)
            if json_obj["num_captions_per_audio"] == 1:
                for item in json_obj["data"]:
                    if "FreeSound" in file and blacklist is not None:
                        if item["id"] in blacklist["FreeSound"]:
                            continue
                    elif ("AudioSet" in file or "AudioCaps" in file) and blacklist is not None:
                        if item["id"] in blacklist["AudioSet"]:
                            continue
                    temp_dict = {"audio": item["audio"], "caption": item["caption"], "id": audio_id,
                                 "duration": item["duration"], "sample_rate": item["sample_rate"], "sample_points": item["sample_points"]}
                    json_data.append(temp_dict)
                    audio_id += 1
            else:
                for item in json_obj["data"]:
                    if "Clotho" in file and blacklist is not None:
                        if item["id"] in blacklist["FreeSound"]:
                            continue
                    if 'valid' in file:
                        chosen_caption = random.randint(1, json_obj["num_captions_per_audio"])
                        temp_dict = {"audio": item["audio"], "caption": item[f"caption_{chosen_caption}"], "id": audio_id,
                                     "duration": item["duration"], "sample_rate": item["sample_rate"], "sample_points": item["sample_points"]}
                        json_data.append(temp_dict)
                    else:
                        for i in range(1, json_obj["num_captions_per_audio"]+1):
                            temp_dict = {"audio": item["audio"], "caption": item[f"caption_{i}"], "id": audio_id,
                                        "duration": item["duration"], "sample_rate": item["sample_rate"], "sample_points": item["sample_points"]}
                            json_data.append(temp_dict)
                    audio_id += 1
    return json_data


class AudioLanguagePretrainDataset(Dataset):

    def __init__(self, json_files, audio_config, blacklist=None):

        self.json_data = _load_json_file(json_files, blacklist)
        self.lengths = [item["duration"] for item in self.json_data]

        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):

        item = self.json_data[index]
        wav_path = item["audio"]
        # duration = item["duration"]
        try:
            # max_seek_time = item['duration'] - self.max_length / self.sr

            orig_tar_len = int(item['sample_rate'] * self.max_length / self.sr) + 10 # load a bit more frames in case convert SR loss
            minmial_len = int(item['sample_rate'] * 5)

            max_start = item['sample_points'] - orig_tar_len
            # start_point = random.randint(0, 10)
            # print(start_point)
            if max_start < 0:
                waveform, sr = soundfile.read(wav_path, start=0, frames=orig_tar_len, dtype=np.float32, always_2d=True)
            else:
                start_point = random.randint(0, max_start)
                # waveform, sr = _av_read(wav_path, seek_time=seek_time, duration=(self.max_length + 10) / self.sr)
                waveform, sr = soundfile.read(wav_path, start=start_point, frames=orig_tar_len, dtype=np.float32, always_2d=True)
            

            # assert waveform.shape[-1] > minmial_len, "audio too short"
            if waveform.shape[0] <= minmial_len: # we replace it with a zero start point
                waveform, sr = soundfile.read(wav_path, start=0, frames=orig_tar_len, dtype=np.float32, always_2d=True)
                
            waveform = torch.from_numpy(waveform).t().contiguous()
            # waveform, _ = librosa.load(wav_path, sr=self.sr, mono=True)
        except Exception as e:
            print(e)
            print(f"Error loading audio file: {wav_path}")
            # assert not 'mtg-jamendo' in wav_path, f'Error loading mtg file: {wav_path}'
            # if 'mtg-jamendo' in wav_path:
            #     print(f'Error loading mtg file: {wav_path}')
            #     if os.path.exists(wav_path):
            #         shutil.move(wav_path, os.path.join('/2214/dongyuanliang/SMC_CodecCLAP/ill_mtg', os.path.basename(wav_path)))
            
            # if 'fma_full' in wav_path:
            #     print(f'Error loading fma file: {wav_path}')
            #     if os.path.exists(wav_path):
            #         shutil.move(wav_path, os.path.join('/2214/dongyuanliang/SMC_CodecCLAP/ill_fma', os.path.basename(wav_path)))
            # if os.path.exists(wav_path):
            #     shutil.move(wav_path, os.path.join('/2214/dongyuanliang/SMC_CodecCLAP/ill_fma', os.path.basename(wav_path)))
            waveform = torch.zeros(self.max_length)
            caption = text_preprocess(item["caption"])
            audio_id = item["id"]
            return waveform, caption, audio_id

        
        # if len(waveform.shape) == 1:
        #     waveform = torch.unsqueeze(waveform, 0)
        waveform = convert_audio(waveform, sr, self.sr, 1)[0]

        if waveform.size(-1) < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.size(-1)))
        elif waveform.size(-1) > self.max_length:
            waveform = waveform[..., :self.max_length]
            

        # if self.max_length != 0:
        #     # if audio length is longer than max_length, we randomly crop it to mac length
        #     if waveform.shape[-1] > self.max_length:
        #         max_start = waveform.shape[-1] - self.max_length
        #         start = random.randint(0, max_start)
        #         waveform = waveform[start: start + self.max_length]

        caption = text_preprocess(item["caption"])
        audio_id = item["id"]

        return waveform, caption, audio_id
        # return duration, caption, audio_id


def collate_fn(batch):
    wav_list = []
    text_list = []
    audio_idx_list = []
    max_length = max([i[0].shape[-1] for i in batch])
    for waveform, text, audio_idx in batch:
        if waveform.shape[-1] < max_length:
            pad_length = max_length - waveform.shape[-1]
            waveform = F.pad(waveform, [0, pad_length], "constant", 0.0)
        wav_list.append(waveform)
        text_list.append(text)
        audio_idx_list.append(audio_idx)

    waveforms = torch.stack(wav_list, dim=0)
    audio_idx = torch.tensor(audio_idx_list).type(torch.long)
    return waveforms, text_list, audio_idx


def pretrain_dataloader(config,
                        bucket: bool = False,
                        bucket_boundaries: tuple = (5, 30, 6),
                        is_distributed: bool = True,
                        num_tasks: int = 0,
                        global_rank: int = 0):
    dataset = AudioLanguagePretrainDataset(config["json_files"], config["audio_args"], config["blacklist"])
    if bucket:
        sampler = BySequenceLengthSampler(lengths=dataset.lengths,
                                          bucket_boundaries=bucket_boundaries,
                                          batch_size=config["data_args"]["batch_size"],
                                          drop_last=True,
                                          seed=config["seed"])
        return DataLoader(dataset=dataset,
                          batch_sampler=BySequenceBatchSampler(sampler, batch_size=config["data_args"]["batch_size"], drop_last=False),
                          shuffle=False,
                          num_workers=config["data_args"]["num_workers"],
                          collate_fn=collate_fn)
    elif is_distributed:
        sampler = DistributedSampler(dataset,
                                     num_replicas=num_tasks,
                                     rank=global_rank,
                                     shuffle=True)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=config["data_args"]["batch_size"],
        num_workers=config["data_args"]["num_workers"],
        pin_memory=False,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_fn,
    )
