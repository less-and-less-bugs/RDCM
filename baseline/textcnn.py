import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from transformers import AutoConfig, AutoModel

"""
TextCNN is for uni-modal classification or just textual feature extractor (according to setting num_classes parameter)
"""


class TextCNN(nn.Module):
    def __init__(self, kernel_sizes, num_filters, num_classes, d_prob,  mode='rand', dataset_name="Pheme"):
        """

        :param kernel_sizes:
        :param num_filters:
        :param num_classes:
        :param d_prob:
        :param mode: rand,roberta-yes,roberta-non, bert-yes, bert-non
        :param path_saved:
        """

        super(TextCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        # roberta-non bert-non bert-yes bert-yes rand
        self.mode = mode
        self.vocab = None
        self.dataset_name = dataset_name
        self.vocab_size = 1000
        self.embedding_dim = 100
        self.embedding = None
        # Bert rand mode need padding_idx, Bert/roberta does not need
        self.load_embeddings()
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # batch_size, sequence_length = x.shape
        # b*l*dim->b*dim*l
        x = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return x.squeeze()

    def load_embeddings(self):
        if self.mode == 'rand':
            if self.dataset_name == "Pheme":
                path_saved = "/data/sunhao/robustfakenews/dataset/vocab/pheme_vocab.pt"
            elif self.dataset_name == "Twitter":
                path_saved = "/data/sunhao/robustfakenews/dataset/vocab/twitter_vocab.pt"
            else:
                print('When Randomly initialized embeddings, the vocabulary is wrong')
                exit(0)
            vocab = torch.load(path_saved)
            self.vocab_size = len(vocab)
            self.embedding_dim = 100
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=vocab['<pad>'])
            self.embedding.weight.data.requires_grad = True
            del vocab
            print('Randomly initialized embeddings are used.')
        else:
            # /data/sunhao/robustfakenews/pretrain_model
            mode = self.mode.split("-")
            assert len(mode) == 2
            path_saved = "/data/sunhao/robustfakenews/pretrain_model"
            if mode[0] == 'roberta':
                config = AutoConfig.from_pretrained(os.path.join(path_saved, "roberta"))
                roberta = AutoModel.from_pretrained(os.path.join(path_saved, "roberta"), config=config)
                weight = roberta.get_input_embeddings().weight
                self.vocab_size = weight.shape[0]
                self.embedding_dim = weight.shape[1]
                self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).from_pretrained(
                    weight)
                # self.embedding.weight.data.copy_(roberta.get_input_embeddings().weight)
                del roberta, config, weight
            elif mode[0] == 'bert':
                config = AutoConfig.from_pretrained(os.path.join(path_saved, "bert"))
                bert = AutoModel.from_pretrained(os.path.join(path_saved, "bert"), config=config)
                weight = bert.get_input_embeddings().weight
                self.vocab_size = weight.shape[0]
                self.embedding_dim = weight.shape[1]
                self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).from_pretrained(weight)
                del bert, config, weight

            else:
                raise ValueError('Unexpected value of mode. Please choose from roberta-non, roberta-yes, rand.')

            if mode[1] == 'non':
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')

            elif mode[1] == 'yes':
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')

            else:
                raise ValueError('Unexpected value of mode[1].')
