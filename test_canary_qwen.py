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
from nemo.collections.speechlm2.models import SALM
from nemo.collections.speechlm2.models.salm import replace_placeholders_and_build_targets
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.parts.pretrained import move_embedding
from nemo.collections.common.prompts import PromptFormatter
from collections import defaultdict
import joblib
from whisper.normalizers import EnglishTextNormalizer
from transformers import GenerationConfig
import random
import jiwer
from utils import parse_args, load_dataset, collate_fn


DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def generate_input_embed(batch, model):
    audios, audio_lens, _, _, context_texts = batch

    input_embeds_final = None
    input_attention_mask = None
    
    if len(audios) == 1:
        assert len(context_texts) == 0
    print(audio_lens)

    # add context audio
    if context_texts:
        for i, (audio, audio_len, context_text) in enumerate(zip(audios[:-1], audio_lens[:-1], context_texts)):
            audio = audio.unsqueeze(0).to(model.device)
            audio_len = torch.tensor([audio_len]).to(model.device)
        
            formatter = PromptFormatter.resolve(model.cfg.prompt_format)(model.tokenizer)
            
            default_template = f"Transcribe the following: {model.audio_locator_tag}"
            # if i == 0 and len(audios) > 1:
            #     content = "I'll provide an example of speech from the same speaker, followed by the correct transcript. Then I'll give you a new audio from the same speaker to transcribe." + default_template
            # else:
            #     content = default_template
            content = default_template
            prompt = [{"role": "user", "content": content}]

            tokens = left_collate_vectors(
                        [formatter.encode_dialog(turns=prompt)["input_ids"]],
                        padding_value=model.text_pad_id,
                    ).to(model.device)
            tokens_to_embed = tokens.where(tokens != model.audio_locator_tag_id, 0).to(model.device)
            token_embeds = model.embed_tokens(tokens_to_embed)

            audio_embeds, audio_embed_lens = model.perception(audio, audio_len)
            audio_embeds = [audio_embeds[j, :elen] for j, elen in enumerate(audio_embed_lens)]

            input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                        input_ids=tokens,
                        embeds=token_embeds,
                        padding_id=model.text_pad_id,
                        placeholder_id=model.audio_locator_tag_id,
                        replacements=audio_embeds,
                        target_ids=None,
                    )
            context_ids = torch.tensor([model.tokenizer.text_to_ids(context_text + " <|im_end|>")]).to(model.device)
            print(context_ids)
            print(context_ids.shape)
            context_token_embeds = model.embed_tokens(context_ids)
            input_embeds = torch.cat((input_embeds, context_token_embeds), dim=1)
            addition_mask = torch.ones((context_token_embeds.size(0), context_token_embeds.size(1))).bool().to(model.device)
            attention_mask = torch.cat((attention_mask, addition_mask), dim=1)
            
            if input_embeds_final is None:
                input_embeds_final = input_embeds
                attention_mask_final = attention_mask
            else:
                input_embeds_final = torch.cat((input_embeds_final, input_embeds), dim=1)
                attention_mask_final = torch.cat((attention_mask_final, attention_mask), dim=1)
    
    # add target audio
    audio = audios[-1].unsqueeze(0).to(model.device)
    audio_len = torch.tensor([audio_lens[-1]]).to(model.device)
    
    default_template = f"Transcribe the following: {model.audio_locator_tag}"
    content = default_template
    prompt = [{"role": "user", "content": content}]

    tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt)["input_ids"]],
                padding_value=model.text_pad_id,
            ).to(model.device)
    tokens_to_embed = tokens.where(tokens != model.audio_locator_tag_id, 0).to(model.device)
    token_embeds = model.embed_tokens(tokens_to_embed)

    audio_embeds, audio_embed_lens = model.perception(audio, audio_len)
    audio_embeds = [audio_embeds[j, :elen] for j, elen in enumerate(audio_embed_lens)]

    input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=model.text_pad_id,
                placeholder_id=model.audio_locator_tag_id,
                replacements=audio_embeds,
                target_ids=None,
            )
    if input_embeds_final is None:
        input_embeds_final = input_embeds
        attention_mask_final = attention_mask
    else:
        input_embeds_final = torch.cat((input_embeds_final, input_embeds), dim=1)
        attention_mask_final = torch.cat((attention_mask_final, attention_mask), dim=1)
    
    return {"inputs_embeds": input_embeds_final, "attention_mask": attention_mask_final}

@torch.no_grad
def generate(model, generation_inputs, generation_config: GenerationConfig = None, **generation_kwargs):
    if generation_config is None:
        generation_config = GenerationConfig(
            bos_token_id=model.text_bos_id,
            eos_token_id=model.text_eos_id,
            pad_token_id=model.text_pad_id,
        )
    with move_embedding(model):
        answer_tokens = model.llm.generate(
            **generation_inputs,
            **generation_kwargs,
            generation_config=generation_config,
        )
    return answer_tokens

def generate_prompt_tokens(model, context_texts=None):
    formatter = PromptFormatter.resolve(model.cfg.prompt_format)(model.tokenizer)
    default_template = f"Transcribe the following: {model.audio_locator_tag}"
    prompt = [{"role": "user", "content": default_template}]
    tokens = left_collate_vectors(
                    [formatter.encode_dialog(turns=prompt)["input_ids"]],
                    padding_value=model.text_pad_id,
                ).to(model.device)
    if context_texts:
        context_ids = []
        context_texts = " ".join(context_texts).strip(".") + ","
        context_ids = model.tokenizer.text_to_ids(context_texts)
        context_ids = torch.tensor([context_ids]).to(model.device)
        tokens = torch.cat((tokens, context_ids), dim=1)
    print(model.tokenizer.ids_to_tokens(tokens[0]))
    tokens = tokens.long()
    return tokens


if __name__ == '__main__':
    args = parse_args()

    # load model
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    model.to(DEVICE)
    model.eval()

    # decoding
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
        
        # audios = audios.to(model.device)
        # audio_lens = audio_lens.to(model.device)
        
        # answer_ids = model.generate(prompts=[[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}]],
        #         audios=audios,
        #         audio_lens=audio_lens,
        #         max_new_tokens=256,)

        # prompts = [[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}]]
        # generation_inputs = generate_input_embed(batch, model)
        # answer_ids = generate(model, generation_inputs=generation_inputs, max_new_tokens=256,)

        prompt_tokens = generate_prompt_tokens(model, context_texts)
        answer_ids = model.generate(prompts=prompt_tokens,
                audios=audios,
                audio_lens=audio_lens,
                max_new_tokens=256,)
        hyps = model.tokenizer.ids_to_text(answer_ids[0].cpu())

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
