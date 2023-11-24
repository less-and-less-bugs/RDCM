import json
from transformers import AutoTokenizer
import torch
import random
import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from .dataset import PhemeSet, TwitterSet
# from dataset import PhemeSet, TwitterSet


def save_file(a, path):
    with open(path, 'w') as f:
        f.write(json.dumps(a))


def read_file(path):
    with open(path, 'r') as f:
        a = json.load(f)
    return a


def compute_mean_std(seeds):
    seeds = np.array(seeds)
    np.mean(seeds[:, 0])
    mean_value = [np.mean(seeds[:, 0]), np.mean(seeds[:, 1]), np.mean(seeds[:, 2]), np.mean(seeds[:, 3])]
    std_value = [np.std(seeds[:, 0]), np.std(seeds[:, 1]), np.std(seeds[:, 2]), np.std(seeds[:, 3])]
    final_value = np.mean(seeds, axis=1)
    final_mean = np.mean(final_value)
    final_std = np.std(final_value)
    return mean_value, std_value, final_mean, final_std

class PadCollate_Pheme:
    def __init__(self, text_dim=0, img_dim=1, label_dim=2, dep_dim=3, type=0, tokenizer_type="bert", vocab_file=None):
        """

        Args:
            text_dim:
            img_dim:
            label_dim:
            dep_dim: the dependencies between words
            type: 0 is without dependency, 1 is with dependency.
            tokenizer_type: different encoding methods, bert, rob
        """
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.label_dim = label_dim
        self.dep_dim = dep_dim
        self.type = type
        self.tokenizer_type = tokenizer_type

        if self.tokenizer_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained('./pretrain_model/bert')
        elif self.tokenizer_type == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('./pretrain_model/roberta')

        else:
            # for textcnn
            self.tokenizer = get_tokenizer("spacy")
            self.vocab = torch.load(vocab_file)
            self.pad_index = self.vocab['<pad>']

    def pad_collate(self, batch):
        """

        Args:
            batch:
        Returns:
            labels: (N), 1 is rumor, 0 is non-rumor.

        """
        texts = list(map(lambda t: t[self.text_dim], batch))
        encoded_texts = []
        if self.tokenizer_type == "bert" or self.tokenizer_type == "roberta":
            encoded_texts = self.tokenizer(texts, is_split_into_words=False, return_tensors="pt", truncation=True,
                                           max_length=100, padding=True)['input_ids']
        else:
            for line in texts:
                words = self.tokenizer(line.strip())
                # truncate
                if len(words) > 100:
                    encoded_texts.append(torch.LongTensor(self.vocab(words[:100])))
                else:
                    encoded_texts.append(torch.LongTensor(self.vocab(words)))
            # pad
            encoded_texts = pad_sequence(encoded_texts, batch_first=True, padding_value=self.pad_index)

        imgs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        imgs = torch.stack(imgs, dim=0)
        labels = list(map(lambda t: t[self.label_dim], batch))
        labels = torch.tensor(labels, dtype=torch.long)
        if self.type == 0:
            return encoded_texts, imgs, labels

    def __call__(self, batch):
        return self.pad_collate(batch)


def construct_edge_text(deps, max_length, chunk=None):
    """

    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    np_mask = []

    for i, dep in enumerate(deps):
        if len(dep) > 3 and len(chunk[i]) > 1:
            dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
            gnn_mask.append(False)
            np_mask.append(True)
            dep_np = torch.cat(dep_np, dim=0).T.contiguous()
        else:
            dep_np = torch.tensor([])
            gnn_mask.append(True)
            np_mask.append(False)
            dep_se.append(dep_np.long())

    np_mask = torch.tensor(np_mask).unsqueeze(1)
    np_mask_ = [torch.tensor(
        [True] * max_length) if gnn_mask[i] else torch.tensor([True] * max_length).index_fill_(0, chunk_,
                                                                                               False).clone().detach()
                for i, chunk_ in enumerate(chunk)]
    np_mask_ = torch.stack(np_mask_)
    np_mask = torch.cat([np_mask_, np_mask], dim=1)
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask, np_mask


def seed_everything(seed: int = 0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_mask_text(seq_len, max_length):
    """

    Args:
        seq_len1(N): list of number of words in a caption without padding in a minibatch
        max_length: the dimension one of shape of embedding of captions of a batch

    Returns:
        mask(N,max_length): Boolean Tensor
    """
    # the realistic max length of sequence
    max_len = max(seq_len)
    if max_len <= max_length:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool), torch.ones(max_length - len, dtype=bool)]) for len in seq_len])
    else:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool),
                        torch.ones(max_length - len, dtype=bool)]) if len <= max_length else torch.zeros(max_length,
                                                                                                         dtype=bool) for
             len in seq_len])

    return mask


def get_metrics(y):
    """
        Computes how accurately model learns correct matching of object with the caption in terms of accuracy

        Args:
            y(N,2): Tensor(cpu). the incongruity score of negataive class, positive class.

        Returns:
            predict_label (list): predict results
    """
    predict_label = (y[:, 0] < y[:, 1]).clone().detach().long().numpy().tolist()
    return predict_label


def get_four_metrics(labels, predicted_labels):
    confusion = metrics.confusion_matrix(labels, predicted_labels)
    # tn, fp, fn, tp
    total = confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]
    acc = (confusion[0][0] + confusion[1][1]) / total
    # about sarcasm
    # if confusion[1][1] == 0:
    #     recall = 0
    #     precision = 0
    # else:
    #     recall = confusion[1][1] / (confusion[1][1] + confusion[1][0])
    #     precision = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    # f1 = 2 * recall * precision / (recall + precision)
    # return acc, recall, precision, f1
    return acc


def get_accuracy(labels: torch.Tensor, y: torch.Tensor):
    # y[:, 0] < y[:, 1] true sarcasm -> 1
    # （N）
    y = y.cpu()
    predict_labels = (y[:, 0] < y[:, 1]).long().numpy()
    labels = labels.cpu().numpy()
    return (labels == predict_labels).sum() / labels.shape[0]


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def l1_norm(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(torch.abs_(a - b)) / a.size(0)


def l2_norm(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(torch.sqrt(torch.sum((a - b) * (a - b), dim=1))) / a.size(0)


def get_three_source_loader(args, train_source_drop=True, train_target_drop=True, test_target_drop=False):
    multi_source_domains_set = []
    multi_source_domains_loader = []
    multi_source_domains_iter = []
    domain_nums = len(args.source)
    if args.data == 'Pheme':
        vocab_path = "./dataset/vocab/pheme_vocab.pt"
    else:
        vocab_path = "./dataset/vocab/twitter_vocab.pt"
    vocab_path_pheme = "./dataset/vocab/pheme_vocab.pt"
    vocab_path_twitter = "./dataset/vocab/twitter_vocab.pt"
    for i in range(domain_nums):
        if args.data == 'Pheme':
            multi_source_domains_set.append(PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                                     img_path="./dataset/pheme/images",
                                                     type=0, events=args.source[i], visual_type='resnet',
                                                     stage='train'))
            multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.workers,
                                                          drop_last=train_source_drop,
                                                          collate_fn=PadCollate_Pheme(type=0,
                                                                                      tokenizer_type=args.tokenizer_type,
                                                                                      vocab_file=vocab_path)))
        elif args.data == 'Twitter':
            multi_source_domains_set.append(TwitterSet(json_path="./dataset/twitter/final_twitter.json",
                                                       img_path="./dataset/twitter/images",
                                                       type=0, events=args.source[i], visual_type='resnet',
                                                       stage='train'))
            multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.workers,
                                                          drop_last=train_source_drop,
                                                          collate_fn=PadCollate_Pheme(type=0,
                                                                                      tokenizer_type=args.tokenizer_type,
                                                                                      vocab_file=vocab_path)))

        elif args.data == 'Cross':
            if args.source == ["charliehebdo", "ottawashooting", "ferguson"]:
                multi_source_domains_set.append(PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                                img_path="./dataset/pheme/images",
                                                type=0, events=args.source[i], visual_type='resnet', stage='train'))
                multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.workers, drop_last=True,
                                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                             vocab_file=
                                                                             vocab_path_pheme)))
            elif args.source == ["sandy", "boston", "sochi"]:
                multi_source_domains_set.append(TwitterSet(
                    json_path="./dataset/twitter/final_twitter.json",
                    img_path="./dataset/twitter/images",
                    type=0, events=args.source[i], visual_type='resnet', stage='train'))
                multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.workers, drop_last=True,
                                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                             vocab_file=
                                                                             vocab_path_twitter)))

        else:
            print("Wrong Source Dataset")
            exit()

        multi_source_domains_iter.append(ForeverDataIterator(multi_source_domains_loader[i]))

    if args.data == 'Pheme':
        train_target_dataset = PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                        img_path="./dataset/pheme/images",
                                        type=0, events=args.target, visual_type='resnet', stage='train')
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers, drop_last=train_target_drop,
                                         collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                     vocab_file=vocab_path))
        test_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers,
                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                             vocab_file=vocab_path), drop_last=test_target_drop)
    elif args.data == 'Twitter':
        train_target_dataset = TwitterSet(json_path="./dataset/twitter/final_twitter.json",
                                          img_path="./dataset/twitter/images",
                                          type=0, events=args.target, visual_type='resnet', stage='train')
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers, drop_last=train_target_drop,
                                         collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                     vocab_file=vocab_path))
        test_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers,
                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                             vocab_file=vocab_path), drop_last=test_target_drop)
    elif args.data == 'Cross':
        if args.source == ["charliehebdo", "ottawashooting", "ferguson"]:
            train_target_dataset = TwitterSet(
                json_path="./dataset/twitter/final_twitter.json",
                img_path="./dataset/twitter/images",
                type=0, events=args.target, visual_type='resnet', stage='train')

            train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_twitter), drop_last=True)
            test_loader =  DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_twitter), drop_last=False)

        elif args.source == ["sandy", "boston", "sochi"]:
            train_target_dataset = PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                            img_path="./dataset/pheme/images",
                                            type=0, events=args.target, visual_type='resnet', stage='train')

            train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_pheme), drop_last=True)

            test_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_pheme), drop_last=False)

    else:
        print("Wrong Target Dataset")
        exit()

    train_target_iter = ForeverDataIterator(train_target_loader)

    print("source domain: {}".format(args.source))
    print("target domain: {}".format(args.target))

    return multi_source_domains_loader, multi_source_domains_iter, train_target_loader, train_target_iter, test_loader


def get_target_source_loader(args, train_source_drop=True, train_target_drop=True, test_target_drop=False):
    multi_source_domains_set = []
    multi_source_domains_loader = []
    multi_source_domains_iter = []
    domain_nums = len(args.source)
    if args.data == 'Pheme':
        vocab_path = "./dataset/vocab/pheme_vocab.pt"
    else:
        vocab_path = "./dataset/vocab/twitter_vocab.pt"
    for i in range(domain_nums):
        if args.data == 'Pheme':
            multi_source_domains_set.append(PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                                     img_path="./dataset/pheme/images",
                                                     type=0, events=args.source[i], visual_type='resnet',
                                                     stage='train'))
        elif args.data == 'Twitter':
            multi_source_domains_set.append(TwitterSet(json_path="./dataset/twitter/final_twitter.json",
                                                       img_path="./dataset/twitter/images",
                                                       type=0, events=args.source[i], visual_type='resnet',
                                                       stage='train'))
        else:
            print("Wrong Source Dataset")
            exit()

        multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                      shuffle=True, num_workers=args.workers,
                                                      drop_last=train_source_drop,
                                                      collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                                  vocab_file=vocab_path)))
        multi_source_domains_iter.append(ForeverDataIterator(multi_source_domains_loader[i]))
    if args.data == 'Pheme':
        train_source_dataset = PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                        img_path="./dataset/pheme/images",
                                        type=0, events=args.source, visual_type='resnet', stage='train')
        test_dataset = PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                        img_path="./dataset/pheme/images",
                                        type=0, events=args.target, visual_type='resnet', stage='train')
    elif args.data == 'Twitter':
        train_source_dataset = TwitterSet(json_path="./dataset/twitter/final_twitter.json",
                                          img_path="./dataset/twitter/images",
                                          type=0, events=args.source, visual_type='resnet', stage='train')
        test_dataset = TwitterSet(json_path="./dataset/twitter/final_twitter.json",
                                          img_path="./dataset/twitter/images",
                                          type=0, events=args.target, visual_type='resnet', stage='train')

    else:
        print("Wrong Target Dataset")
        exit()

    train_target_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                                  vocab_file=vocab_path), drop_last=True)
    train_target_iter = ForeverDataIterator(train_target_loader)
    test_loader = DataLoader(test_dataset , batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                                  vocab_file=vocab_path), drop_last=False)
    print("source domain: {}".format(args.source))
    print("target domain: {}".format(args.target))

    return multi_source_domains_loader, multi_source_domains_iter, train_target_loader, train_target_iter, test_loader