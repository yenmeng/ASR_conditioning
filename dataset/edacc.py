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
                spk_id = row['speaker'][0]
                accent = row['accent'][0]
                fid = row['audio'][0]['path'].replace('.wav', '')
                audio = row['audio'][0]['bytes']
                audio_duration = self.wav_duration_from_bytes(audio)
                if audio_duration > 40:
                    continue # skip too long
                text = self._filter(row['text'][0])
                if len(text.split()) < 2:
                    continue # skip empty transcript
                pqs.append((audio, audio_duration, text, spk_id, accent))

        self.dataset = []
        for i, item in enumerate(tqdm(pqs)):
            audio, audio_duration, text, spk_id, accent = item
            audios = []
            context_text_normalized = []
            if context:
                if context_type == "random":
                    inds = list(range(len(pqs)))
                    inds.remove(i)
                    total_duration = 0
                    while total_duration < 20 and len(inds) > 0:
                        context_id = random.choice(inds)
                        context_audio, context_audio_duration, context_text, _, _ = pqs[context_id]
                        if round(total_duration + context_audio_duration) <= 20:
                            context_text_normalized.append(context_text + ".")
                            audios.append(context_audio)
                            total_duration += context_audio_duration
                            inds.remove(context_id)
                        else:
                            if total_duration == 0: # still empty, continue sampling
                                inds.remove(context_id)
                                continue
                            else:
                                break

                elif context_type == "speaker":
                    inds = [ind for ind, v in enumerate(pqs) if v[3] == spk_id and ind != i]
                    total_duration = 0
                    while total_duration < 10 and len(inds) > 0:
                        context_id = random.choice(inds)
                        context_audio, context_audio_duration, context_text, _, _ = pqs[context_id]
                        if round(total_duration + context_audio_duration) <= 10:
                            context_text_normalized.append(context_text + ".")
                            audios.append(context_audio)
                            total_duration += context_audio_duration
                            inds.remove(context_id)
                        else:
                            if total_duration == 0: # still empty, continue sampling
                                inds.remove(context_id)
                                continue
                            else:
                                break

                elif context_type == "same":
                    # context_audio = audio
                    audios.append(audio)
                    context_text_normalized.append(text + ".")
                else:
                    raise NotImplementedError
            
            audios.append(audio) # put target audio at the end
            self.dataset.append((audios, text, f"{spk_id}_{i}", context_text_normalized))

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
        wav = torch.from_numpy(wav)
        return wav, sr
    
    def _resample(self, wav, source_sr, target_sr=16000):
        resampler = torchaudio.transforms.Resample(source_sr, target_sr, dtype=wav.dtype)
        resampled_wav = resampler(wav)
        return resampled_wav, target_sr
    
    def _filter(self, text):
        # replace tags
        text = text.replace('IGNORE_TIME_SEGMENT_IN_SCORING', '')
        text = re.sub(r"(<\w\S+>)", "", text) 
        text = text.strip().capitalize()
        tokens = text.split()
        for i in range(len(tokens)):
            if tokens[i] == "i":
                tokens[i] = "I"
            elif tokens[i] == "i'm":
                tokens[i] = "I'm"
        text = " ".join(tokens)
        return text
    
    def __getitem__(self, item):
        audios, text, fid, context_text = self.dataset[item]
        loaded_audios = []
        # load context audios
        if context_text:
            assert len(audios) > 1
            for i in range(len(audios) - 1):
                context_audio, source_sr = self._read_bytes_to_arr(audios[i])
                context_audio, sample_rate = self._resample(context_audio, source_sr)
                assert sample_rate == 16000
                loaded_audios.append(context_audio.squeeze())
        # load target audio
        target_audio, source_sr = self._read_bytes_to_arr(audios[-1])
        target_audio, sample_rate = self._resample(target_audio, source_sr)
        assert sample_rate == 16000
        loaded_audios.append(target_audio.squeeze()) # put the target audio at the end

        audio_len = [len(a) for a in loaded_audios]

        return loaded_audios, audio_len, text, fid, context_text

    def __len__(self):
        return len(self.dataset)
