import os
import random
import re
from glob import glob
import json
import torch
import torchaudio
import whisper
from tqdm import tqdm
from typing import List
# from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_knn(query, corpus, dist_func="euclidean"):
    if dist_func == "euclidean":
        d = torch.cdist(query, corpus, p=2)
        print(d.shape)
        B, N = d.shape
        ar = torch.arange(B, device=d.device)
        d[ar, ar] = float("inf")

        vals, idx = torch.topk(d, 1, dim=-1, largest=False)
        print(idx)
        return idx
    elif dist_func == "cosine":
        pass
    else:
        raise NotImplementedError

class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, root='/disk/scratch/s2522924/LibriSpeech', split="test-clean", context=False, context_type='random'):
        
        match = re.search(r'(?P<base_split>[a-z]+-[a-z]+)_?(?P<snr>-?\d+db)?-?(?P<noise_type>[mgr])?', split)
        base_split = match.group('base_split')
        snr = match.group('snr')
        noise_type = match.group('noise_type')
        
        file_list = sorted(glob(os.path.join(root, split, "**/*.flac"), recursive=True))
        # base_file_list = [fname.replace(split, base_split+'_0db-g') for fname in file_list]
        base_file_list = file_list
        trans_list = sorted(glob(os.path.join(root, base_split, "**/*.trans.txt"), recursive=True))
        self.label_dict = {}
        
        # parse label
        for trans in trans_list:
            with open(trans, 'r') as transfile:
                for l in transfile:
                    fid, text = l.split(' ', 1)
                    self.label_dict[fid] = self._filter(text) 
        # get audio durations
        with open(os.path.join(root, f"{base_split}.json")) as f:
            self.duration_dict = json.load(f)
        
        if context:
            if context_type == "text_sim":
                tokenized_corpus = [t.split(" ") for t in list(self.label_dict.values())]
                # tokenized_corpus = [t.split(" ") for t in list(query_text.values())]
                # bm25 = BM25Okapi(tokenized_corpus)

                vectorizer = TfidfVectorizer(stop_words='english')
                corpus = vectorizer.fit_transform(self.label_dict.values())

        print('collecting audio...')
        self.dataset = []
        for i in tqdm(range(len(file_list))):
            fid = file_list[i].split('/')[-1].split('.')[0]
            # audio, sample_rate = torchaudio.load(file_list[i])
            # if len(audio.squeeze()) / sample_rate > 20: 
            #     continue
            # audio = audio.squeeze()
            text = self.label_dict[fid] 
            
            audios = []
            context_text_normalized = []
            if context:
                # load context
                if context_type == "random":

                    inds = list(range(len(base_file_list)))
                    inds.remove(i)

                    context_id = random.choice(inds)
                    context_file = base_file_list[context_id]
                    context_text = self.label_dict[context_file.split('/')[-1].split('.')[0]] 

                    audios.append(context_file)
                    context_text_normalized.append((context_text + '.'))

                elif context_type == "speaker":
                    spk_id = fid.split('-')[0]
                    inds = [ind for ind, v in enumerate(base_file_list) if spk_id in v and ind != i]
                    total_duration = 0
                    while total_duration < 20:
                        context_id = random.choice(inds)
                        context_file = base_file_list[context_id]
                        context_fid = context_file.split('/')[-1].split('.')[0]
                        context_text = self.label_dict[context_fid] 
                        context_audio_duration = self.duration_dict[context_fid]
                        if round(total_duration + context_audio_duration) <= 20:
                            audios.append(context_file)
                            context_text_normalized.append((context_text + '.'))
                            total_duration += context_audio_duration
                            inds.remove(context_id)
                        else:
                            if total_duration == 0:
                                inds.remove(context_id)
                                continue
                            else:
                                break

                        
                elif context_type == "same":
                    context_file = base_file_list[i]
                    context_text = self.label_dict[context_file.split('/')[-1].split('.')[0]]

                    audios.append(context_file)
                    context_text_normalized.append((context_text + '.'))

                elif context_type == "text_sim":
                    # # bm25
                    # tokenized_query = text.split(" ")
                    # tokenized_query = query_text[fid].split(" ")
                    # scores = bm25.get_scores(tokenized_query)
                    # _, candidates = torch.topk(torch.tensor(scores), k=5) 
                    # # tf-idf
                    query = vectorizer.transform([text]) 
                    # query = vectorizer.transform([query_text[fid]]) 
                    scores = cosine_similarity(query, corpus).flatten()
                    _, candidates = torch.topk(torch.tensor(scores), k=1) 
                    
                    context_id = candidates[0]

                    if context_id is not None:
                        audios.append(context_file)
                        context_text_normalized.append((context_text + '.'))
                else:
                    raise NotImplementedError
            
            audios.append(file_list[i]) # put target audio at the end
            self.dataset.append((audios, text, fid, context_text_normalized)) 
    
    def _filter(self, text):
        text = text.strip('\n').strip().capitalize()
        tokens = text.split()
        for i in range(len(tokens)):
            if tokens[i] == "i":
                tokens[i] = "I"
            elif tokens[i] == "i'm":
                tokens[i] = "I'm"
        text = " ".join(tokens)
        return text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audios, text, fid, context_text = self.dataset[item]
        loaded_audios = []
        # load context audios
        if context_text:
            assert len(audios) > 1
            for i in range(len(audios) - 1):
                context_audio, sample_rate = torchaudio.load(audios[i])
                assert sample_rate == 16000
                loaded_audios.append(context_audio.squeeze())
        # load target audio
        target_audio, sample_rate = torchaudio.load(audios[-1])
        assert sample_rate == 16000
        loaded_audios.append(target_audio.squeeze()) # put the target audio at the end

        audio_len = [len(a) for a in loaded_audios]

        return loaded_audios, audio_len, text, fid, context_text
