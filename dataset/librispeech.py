import os
import random
import re
from glob import glob
import torch
import torchaudio
import whisper
from tqdm import tqdm
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
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
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
                    self.label_dict[fid] = text.strip('\n')
        
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
            audio, sample_rate = torchaudio.load(file_list[i])
            # if len(audio.squeeze()) / sample_rate > 20: 
            #     continue
            audio = audio.squeeze()

            text = self.label_dict[fid] 
            
            if not context:
                self.dataset.append((audio, sample_rate, text, fid, None))
            
            else:
                # load context
                if context_type == "random":

                    inds = list(range(len(base_file_list)))
                    inds.remove(i)

                    context_id = random.choice(inds)
                    context_file = base_file_list[context_id]
                    context_audio, sample_rate = torchaudio.load(context_file)
                    context_text = self.label_dict[context_file.split('/')[-1].split('.')[0]] 
                    # while len(torch.cat((context_audio.squeeze(), audio.squeeze()))) / sample_rate > 30:
                    #     inds.remove(context_id)
                    #     if not inds:
                    #         context_id = None
                    #         break
                    #     context_id = random.choice(inds)
                    #     context_file = base_file_list[context_id]
                    #     context_audio, sample_rate = torchaudio.load(context_file)
                    #     context_text = self.label_dict[context_file.split('/')[-1].split('.')[0]] 
                    #     # try:
                    #     #     context_text = query_text[context_file.split('/')[-1].split('.')[0]] 
                    #     # except:
                    #     #     continue
                    # print(file_list[i], context_file)
                    
                    if context_id is not None:
                        audio = torch.cat((context_audio.squeeze(), audio.squeeze()))
                        context_text_normalized = (context_text + '.').capitalize()
                        self.dataset.append((audio, sample_rate, text, fid, context_text_normalized)) 
                    else:
                        self.dataset.append((audio, sample_rate, text, fid, None))
                        
                elif context_type == "same":
                    context_file = base_file_list[i]
                    context_audio, sample_rate = torchaudio.load(context_file)
                    context_text = self.label_dict[context_file.split('/')[-1].split('.')[0]]

                    audio = torch.cat((context_audio.squeeze(), audio.squeeze()))
                    context_text_normalized = (context_text + '.').capitalize()
                    self.dataset.append((audio, sample_rate, text, fid, context_text_normalized)) 

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
                        # print(file_list[i], context_file)
                        audio = torch.cat((context_audio.squeeze(), audio.squeeze()))
                        context_text_normalized = (context_text + '.').capitalize()
                        self.dataset.append((audio, sample_rate, text, fid, context_text_normalized)) 
                    else:
                        self.dataset.append((audio, sample_rate, text, fid, None))

                else:
                    raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, fid, context_text = self.dataset[item]
        assert sample_rate == 16000
        audio_len = len(audio)

        return audio, audio_len, text, fid, context_text
