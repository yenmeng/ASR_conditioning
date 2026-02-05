import os
os.environ["HF_HOME"] = "/home/s2522924/.cache/huggingface/"
import re
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torchaudio
from nemo.collections.asr.models import EncDecMultiTaskModel
from collections import defaultdict
import joblib
from whisper.normalizers import EnglishTextNormalizer
import random
import jiwer
from utils import parse_args, load_dataset, collate_fn

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def decode(model, input_signal, input_signal_length, transcript, transcript_length, prompt_length, max_len=256):
    hyp = ''
    for _ in range(max_len):
        with torch.no_grad():
            res = model(input_signal=input_signal, 
                        input_signal_length=input_signal_length,
                        transcript=transcript,
                        transcript_length=transcript_length
                        )
        tokens = res[0].argmax(dim=-1)
        # print(tokens)
        if tokens[-1][-1] == tokenizer.eos_id:
            return hyp
        transcript = torch.cat((transcript, tokens[:, -1:]), dim=-1)
        transcript_length = torch.tensor([transcript.shape[1]]).to(model.device)
        hyp = tokenizer.ids_to_text(transcript[-1][prompt_length:])
    
    return hyp

if __name__ == '__main__':
    args = parse_args()

    # load model
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b', map_location='cpu')
    model.to(DEVICE)
    model.eval()
    print(model.device)
        
    # update decode params
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    tokenizer = model.tokenizer

    normalizer = EnglishTextNormalizer()

    # dataset = eval(DATASET_DICT[args.dataset])(split=args.split, context=args.context, context_type=args.context_type)
    dataset = load_dataset(args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    print(args.split)
    hypotheses = []
    references = []

    preds = []
    gts = []
    for i, batch in enumerate(tqdm(loader)):
        audios, audio_lens, texts, fids, context_texts = batch
        
        audios = audios.to(model.device)
        audio_lens = torch.tensor(audio_lens).to(model.device)

        special_tokens = ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|en|>', '<|pnc|>']
        # special_tokens = ['<|startoftranscript|>', '<|emo:undefined|>', '<|en|>', '<|en|>', '<|pnc|>', '<|noitn|>', '<|notimestamp|>', '<|nodiarize|>']
        special_token_ids = tokenizer.tokens_to_ids(special_tokens, ['spl_tokens' for _ in range(len(special_tokens))])

        if args.context:
            input_token_ids = tokenizer.text_to_ids(context_texts, 'en')
            print(input_token_ids)
            print(len(input_token_ids))
            transcript = special_token_ids + input_token_ids
            # transcript = special_token_ids + input_token_ids + [tokenizer.eos_id] + special_token_ids
        else:
            transcript = special_token_ids
        
        transcript = torch.tensor(transcript).unsqueeze(0).to(model.device)
        prompt_length = torch.tensor([transcript.shape[1]]).to(model.device)
        transcript_length = prompt_length

        with torch.no_grad():
            hyps = decode(model, audios, audio_lens, transcript, transcript_length, prompt_length)
        
        print(f"context: {context_texts}")
        print(f"hyp: {hyps}")
        print(f"ref: {texts}")

        if normalizer(texts):
            hypotheses.extend([hyps])
            references.extend([texts])

            preds.append((fids[0], normalizer(hyps)))
            gts.append((fids[0], normalizer(texts)))

    data = {}
    data["hypotheses"] = [normalizer(h) for h in hypotheses]
    data["references"] = [normalizer(r) for r in references]

    for i, (hyp, ref) in enumerate(zip(data["hypotheses"], data["references"])):
        if i == 20:
            break
        print(f"hyp: {hyp}")
        print(f"ref: {ref}")

    wer = jiwer.wer(list(data["references"]), list(data["hypotheses"]))
    cer = jiwer.cer(list(data["references"]), list(data["hypotheses"]))
    print(f"WER: {wer * 100:.2f} %")
    print(f"CER: {cer * 100:.2f} %")

    if args.save_pred:
        if args.context:
            file_name = f'transcript_{args.split}_{args.context_type}_hyps.txt'
        else:
            file_name = f'transcript_{args.split}_hyps.txt'
        with open(os.path.join(args.save_dir, file_name), 'w') as f:
            f.write("\n".join([f"{item[0]} {item[1]}" for item in preds]))
        f.close()
        if gts:
            with open(os.path.join(args.save_dir, f'transcript_{args.split}_refs.txt'), 'w') as f:
                f.write("\n".join([f"{item[0]} {item[1]}" for item in gts]))
            f.close()
