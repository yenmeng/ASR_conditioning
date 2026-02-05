import argparse
import string
import num2words

from dataset.librispeech import LibriSpeech
from dataset.edacc import EdAcc
from dataset.l2artic import L2Artic

DATASET_DICT = {"librispeech": "LibriSpeech", "edacc": "EdAcc", "l2artic": "L2Artic"}

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for inference")
    parser.add_argument('--dataset', type=str, default='librispeech', choices=["librispeech", "edacc", "l2artic"])
    parser.add_argument('--split', type=str, default='test-clean')
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--context_type', type=str, default="random", choices=["random", "speaker", "accent", "same", "text_sim", "audio_sim"])
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--save_pred', action='store_true')

    return parser.parse_args()

def collate_fn(batch):
    audios, audio_lens, texts, fids, context_texts = list(zip(*batch))
    audios = audios[0].unsqueeze(0)
    texts = texts[0]
    context_texts = context_texts[0]
    return audios, audio_lens, texts, fids, context_texts

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    normalized_text = []
    for wrd in text.split():
        if wrd.isdigit():
            wrd = num2words(int(wrd))
        normalized_text.append(wrd.strip(string.punctuation))

    text = ' '.join(normalized_text)
    return text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))

def load_dataset(args):
    return eval(DATASET_DICT[args.dataset.lower()])(split=args.split, context=args.context, context_type=args.context_type)
