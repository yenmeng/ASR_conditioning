import os
import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import re
import polars as pl
from tqdm import tqdm
from torch.utils.data import DataLoader
import io
import struct
import soundfile as sf
import random

SAMPLE_RATE = 16000 

class EdAcc(Dataset):
    def __init__(self, root='/disk/scratch/s2522924/edacc', split="test", context=False, context_type='random'):
        
        files = sorted(glob.glob(os.path.join(root, f"**/{split}-*.parquet"), recursive=True))

        self.silence_seg = torch.zeros(int(SAMPLE_RATE * 0.4), dtype=torch.float32)

        pqs = []
        for f in tqdm(files):
            pq = pl.read_parquet(f)

            for i in tqdm(range(len(pq))):
                row = pq[i]
                accent = row['accent'][0]
                spk_id = row['speaker'][0]
                fid = row['audio'][0]['path'].replace('.wav', '')
                # audio, source_sr = self._read_bytes_to_arr(row['audio'][0]['bytes'])
                # audio, sample_rate = self._resample(audio, source_sr)
                # if len(audio) / sample_rate > 40:
                #     continue # skip too long
                audio = row['audio'][0]['bytes']
                audio_len = self.wav_duration_from_bytes(audio)
                if audio_len > 30:
                    continue
                text = self._filter(row['text'][0])
                if len(text.split()) < 2:
                    continue # skip empty transcript
                pqs.append((audio, audio_len, text, spk_id, accent))
        # pqs = sorted(pqs, key=lambda x: x[1], reverse=True)

        self.dataset = []
        for i, item in enumerate(tqdm(pqs)):
            audio, audio_len, text, spk_id, accent = item
            if not context:
                self.dataset.append((audio, text, f"{spk_id}_{i}", None, None))
            else:
                if context_type == "random":
                    inds = np.arange(len(pqs))
                    inds.remove(i)
                    context_id = random.choice(inds)
                    context_audio, _, context_text, _, _ = pqs[context_id]
                    context_text_normalized = (context_text + ".").capitalize()
                elif context_type == "speaker":
                    inds = [(ind, v[1]) for ind, v in enumerate(pqs) if v[3] == spk_id and ind != i]
                    inds = sorted(inds, key=lambda x: x[1], reverse=True)
                    # context_id = random.choice(range(0, min(len(inds), 5)))
                    # context_id = inds[context_id][0]
                    context_id = inds[0][0]
                    context_audio, _, context_text, _, _ = pqs[context_id]
                    context_text_normalized = (context_text + ".").capitalize()
                elif context_type == "accent":
                    inds = [(ind, v[1]) for ind, v in enumerate(pqs) if v[-1] == accent and ind != i]
                    inds = sorted(inds, key=lambda x: x[1], reverse=True)
                    context_id = inds[0][0]
                    context_audio, _, context_text, _, _ = pqs[context_id]
                    context_text_normalized = (context_text + ".").capitalize()
                elif context_type == "same":
                    context_audio = audio
                    context_text_normalized = (text + ".").capitalize()

                self.dataset.append((audio, text, f"{spk_id}_{i}", context_audio, context_text_normalized)) 
        del pqs
    
    def wav_duration_from_bytes(self, b: bytes) -> float:
        f = io.BytesIO(b)
        f.seek(22)
        channels = struct.unpack("<H", f.read(2))[0]
        f.seek(24)
        sample_rate = struct.unpack("<I", f.read(4))[0]
        f.seek(34)
        bits_per_sample = struct.unpack("<H", f.read(2))[0]
        f.seek(40)
        data_size = struct.unpack("<I", f.read(4))[0]
        bytes_per_sample = bits_per_sample // 8
        duration = data_size / (sample_rate * channels * bytes_per_sample)
        return duration

    def _read_bytes_to_arr(self, bytes):
        wav, sr = sf.read(io.BytesIO(bytes), dtype='float32')
        return torch.from_numpy(wav), sr
    
    def _resample(self, wav, source_sr, target_sr=16000):
        resampler = torchaudio.transforms.Resample(source_sr, target_sr, dtype=wav.dtype)
        resampled_wav = resampler(wav)
        return resampled_wav, target_sr

    def _filter(self, text):
        # replace tags
        text = text.replace('IGNORE_TIME_SEGMENT_IN_SCORING', '')
        text = re.sub(r"(<\w\S+>)", "", text) 
        text = text.strip()
        return text
    
    def __getitem__(self, item):
        audio, text, fid, context_audio, context_text = self.dataset[item]
        audio, source_sr = self._read_bytes_to_arr(audio)
        audio, sample_rate = self._resample(audio, source_sr)
        if context_audio is not None:
            context_audio, source_sr = self._read_bytes_to_arr(context_audio)
            context_audio, sample_rate = self._resample(context_audio, source_sr)
            audio = torch.cat((context_audio.squeeze(), audio.squeeze()))
        assert sample_rate == 16000
        audio_len = len(audio)

        return audio, audio_len, text, fid, context_text

    def __len__(self):
        return len(self.dataset)
