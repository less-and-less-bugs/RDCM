import torch
from utils import PhemeSet, TwitterSet
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

"""
This is for extracting vocab for both dataset 
"""


if __name__ == '__main__':
    twitter_set = TwitterSet(json_path="../final_twitter.json", img_path="../twitter/images",
                             type=0, events=["sandy", "boston", "sochi", "malaysia"], visual_type='resnet',
                             stage='train')

    twitter_vocab_path = "../vocab/twitter_vocab.pt"
    pheme_vocab_path = "../vocab/pheme_vocab.pt"

    print(twitter_set[0][0])
    tokenizer = get_tokenizer("spacy")
    lines = []
    for i in range(len(twitter_set)):
        lines.append(tokenizer(twitter_set[i][0].strip()))
    line_iter = iter(lines)
    vocab = build_vocab_from_iterator(line_iter, specials=["<unk>", '<pad>'], min_freq=5)
    vocab.set_default_index(vocab['<unk>'])
    print(len(vocab))
    # torch.save(vocab, pheme_vocab_path)
    torch.save(vocab, twitter_vocab_path)

