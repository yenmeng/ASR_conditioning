import os
os.environ["HF_HOME"] = "/home/s2522924/.cache/huggingface/"
import re
import datetime
import time
import json
import numpy as np
from tqdm import tqdm
import torch
import torchaudio
from collections import defaultdict
import joblib
from whisper.normalizers import EnglishTextNormalizer
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import random
import jiwer
from utils import parse_args, load_dataset, collate_fn


DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


def generate_prompt(audios, context_texts=None):
    # Define prompt structure
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    # if context_texts:
    #     assert len(audios) == len(context_texts) + 1

    # num_shots = len(audios) - 1
    # if num_shots == 0:
    #     prompt = f"{user_prompt}<|audio_1|>Transcribe the audio clip into text.{prompt_suffix}{assistant_prompt}"
    
    # else:
    #     prompt = f"{user_prompt}I'll provide {'an example' if num_shots == 1 else f'{num_shots} examples'} from a non-native English speaker, followed by the correct transcription. Then I'll give you a new audio from the same speaker to transcribe.{prompt_suffix}{assistant_prompt}I understand. I'll listen to the {'example' if num_shots == 1 else 'examples'} and use {'it' if num_shots == 1 else 'them'} to accurately transcribe the final audio.{prompt_suffix}"
    #     for i, text in enumerate(context_texts):
    #         prompt += f"{user_prompt}<|audio_{i+1}|>Transcribe this audio:{prompt_suffix}{assistant_prompt}{text}{prompt_suffix}"
    #     prompt += f"{user_prompt}<|audio_{len(audios)}|>Please transcribe this audio from the same speaker:{prompt_suffix}{assistant_prompt}"
    # print(prompt)

    speech_prompt = "Transcribe the audio clip into text."
    prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
    
    if context_texts:
        prompt += f'{" ".join(context_texts).strip(".") + ","}'
    print(prompt)

    return prompt


if __name__ == '__main__':
    args = parse_args()

    # Define model path
    model_path = "microsoft/Phi-4-multimodal-instruct"

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir="/home/s2522924/.cache/huggingface")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
        # if you do not use Ampere or later GPUs, change attention to "eager"
        _attn_implementation='eager',
        cache_dir="/home/s2522924/.cache/huggingface"
    ).cuda()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)

    normalizer = EnglishTextNormalizer()

    dataset = load_dataset(args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    print(args.split)
    hypotheses = []
    references = []

    preds = []
    gts = []
    for i, batch in enumerate(tqdm(loader)):
        audios, audio_lens, texts, fids, context_texts = batch
        
        # audios = audios.to(model.device)
        # audio_lens = audio_lens.to(model.device)
        
        prompt = generate_prompt(audios, context_texts)
        inputs = processor(text=prompt, audios=[(a.squeeze(0), 16000) for a in audios], return_tensors='pt').to('cuda:0')
        # inputs = processor(text=prompt, audios=[(audios.squeeze(0), 16000)], return_tensors='pt').to('cuda:0')

        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        hyps = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
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
