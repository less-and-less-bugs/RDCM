import torch
import torch.nn as nn
from transformers import BertModel
import torchvision
import torch.nn.functional as F
from baseline import TextCNN
from copy import deepcopy




"""
Use layer norm instead of Batch Norm 
Encoder is for extracting textual using Bert, visual using resnet, cls (use as contrastive learning measurement) and
the bert is finetuned wholly and resent is finetuned by setting layers 

Encoder_TextCnn is for extracting textual using TextCnn, visual using Resnet, cls (use as contrastive learning measurement)
and resent is finetuned by setting layers 

DRJModel is bert + resnet， return train_texts, train_imgs, train_y, instance_cls
DRJModel_Mgat is bert + resnet + domain adaptive selection， return train_texts, train_imgs, train_y, instance_cls 
DRJModel_TexCnn is TextCNN + resnet， return train_texts, train_imgs, train_y, instance_cls + domain adaptive selection 
(can  )

"""


# for this code implementation, we visualize the embedding of different domains at the beginning of each epoch


class Encoder(nn.Module):
    def __init__(self, input_size=768, out_size=300, freeze_id=-1, droprate=0.2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = torchvision.models.resnet50(pretrained=True)
        # self.visual_encoder = torchvision.models.resnet101(pretrained=True)
        # remove classification layer
        self.droprate = droprate
        self.instance_discriminator = torchvision.models.resnet50(pretrained=True)

        # self.instance_discriminator = self.visual_encoder.fc
        self.visual_encoder = torch.nn.Sequential(*(list(self.visual_encoder.children())[:-1]))
        self.text_linear = nn.Sequential(nn.Linear(in_features=768, out_features=self.input_size), nn.ReLU(),
                                         # nn.Dropout(p=0.2),
                                         nn.Linear(in_features=self.input_size, out_features=self.out_size),
                                         nn.LayerNorm(self.out_size),
                                         nn.ReLU(),
                                         nn.Dropout(self.droprate)
                                         )
        self.img_linear = nn.Sequential(nn.Linear(in_features=2048, out_features=1024), nn.ReLU(),
                                        # nn.Dropout(p=0.2),
                                        nn.Linear(in_features=1024, out_features=self.out_size),
                                        nn.LayerNorm(self.out_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.droprate)
                                        )
        # id = 8 freeze all parameter
        # id=7 freeze the last
        # id=-1 do not freeze
        self.freeze_layer(self.visual_encoder, freeze_id)
        for para in self.instance_discriminator.parameters():
            para.requires_grad = False

    def forward(self, texts, imgs):
        # texts = self.bert_model(**texts)[0]
        instance_cls = self.instance_discriminator(imgs)
        imgs = self.visual_encoder(imgs)
        texts = self.bert_model(**texts)[0]
        # remove cls token and sep token
        # texts = texts[:, 1:-1, :]
        # only use [cls] token for classification
        imgs = imgs.squeeze()
        # instance_cls = self.instance_discriminator(imgs)
        texts = texts[:, 0, :]
        texts = self.text_linear(texts)
        imgs = self.img_linear(imgs)
        return texts, imgs, instance_cls

    def get_parameters(self):
        bert_params = list(map(id, self.bert_model.parameters()))
        conv_params = list(map(id, self.visual_encoder.parameters()))
        params = [
            {"params": self.bert_model.parameters(), "lr": None},
            {"params": self.visual_encoder.parameters(), "lr": None},
            {"params": filter(lambda p: id(p) not in bert_params + conv_params, self.parameters()), "lr": None},
        ]
        return params

    def freeze_layer(self, model, freeze_layer_ids):
        count = 0
        para_optim = []
        for k in model.children():
            # 6 should be changed properly
            if count > freeze_layer_ids:  # 6:
                for param in k.parameters():
                    para_optim.append(param)
            else:
                for param in k.parameters():
                    param.requires_grad = False
            count += 1
        # print count
        return para_optim


class Encoder_TextCnn(nn.Module):
    def __init__(self, out_size=300, freeze_id=-1, d_prob=0.3,
                 kernel_sizes=[3, 4, 5], num_filters=100, mode='rand', dataset_name="Pheme"):
        super(Encoder_TextCnn, self).__init__()
        self.out_size = out_size
        self.droprate = d_prob
        self.mode = mode
        self.dataset_name = dataset_name
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters

        self.visual_encoder = torchvision.models.resnet50(pretrained=True)
        # self.visual_encoder = torchvision.models.resnet101(pretrained=True)
        # remove classification layer
        # kernel_sizes, num_filters, num_classes, d_prob, mode = 'rand-', dataset_name = "Pheme"
        self.textcnn = TextCNN(kernel_sizes=self.kernel_sizes,
                               num_filters=self.num_filters,
                               num_classes=self.out_size, d_prob=self.droprate, mode=self.mode,
                               dataset_name=self.dataset_name
                               )
        self.instance_discriminator = torchvision.models.resnet50(pretrained=True)

        # self.instance_discriminator = self.visual_encoder.fc
        self.visual_encoder = torch.nn.Sequential(*(list(self.visual_encoder.children())[:-1]))
        self.text_linear = nn.Sequential(nn.LayerNorm(self.out_size))
        self.img_linear = nn.Sequential(nn.Linear(in_features=2048, out_features=1024), nn.ReLU(),
                                        # nn.Dropout(p=0.2),
                                        nn.Linear(in_features=1024, out_features=self.out_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.droprate),
                                        nn.LayerNorm(self.out_size)
                                        )
        # id = 8 freeze all parameter
        # id=7 freeze the last
        # id=-1 do not freeze
        self.freeze_layer(self.visual_encoder, freeze_id)
        for para in self.instance_discriminator.parameters():
            para.requires_grad = False

    def forward(self, texts, imgs):
        # texts = self.bert_model(**texts)[0]
        instance_cls = self.instance_discriminator(imgs)
        imgs = self.visual_encoder(imgs)
        texts = self.textcnn(texts)
        # remove cls token and sep token
        # texts = texts[:, 1:-1, :]
        # only use [cls] token for classification
        imgs = imgs.squeeze()
        # instance_cls = self.instance_discriminator(imgs)
        texts = self.text_linear(texts)
        imgs = self.img_linear(imgs)
        return texts, imgs, instance_cls

    def get_parameters(self):
        textcnn_params = list(map(id, self.textcnn.parameters()))
        conv_params = list(map(id, self.visual_encoder.parameters()))
        params = [
            {"params": self.textcnn.parameters(), "lr": None},
            {"params": self.visual_encoder.parameters(), "lr": None},
            {"params": filter(lambda p: id(p) not in textcnn_params + conv_params, self.parameters()), "lr": None},
        ]
        return params

    def freeze_layer(self, model, freeze_layer_ids):
        count = 0
        para_optim = []
        for k in model.children():
            # 6 should be changed properly
            if count > freeze_layer_ids:  # 6:
                for param in k.parameters():
                    para_optim.append(param)
            else:
                for param in k.parameters():
                    param.requires_grad = False
            count += 1
        # print count
        return para_optim


class DRJModel(nn.Module):
    def __init__(self, input_size=768, out_size=300, num_label=2, freeze_id=-1):
        super(DRJModel, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_label = num_label
        self.encoder = Encoder(input_size=self.input_size, out_size=self.out_size, freeze_id=freeze_id)
        # self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=self.out_size), nn.ReLU(),
        #                                nn.Linear(in_features=self.out_size, out_features=int(self.out_size / 2)))
        # the below cls_layer is for multimodal baseline
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=self.out_size), nn.ReLU())
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, train_texts, train_imgs):
        """
        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))
        # train_texts*train_imgs
        # train_y = self.head(torch.cat((train_texts, f), dim=1))
        train_y = self.head(f)
        return train_texts, train_imgs, train_y, instance_cls

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = self.encoder.get_parameters()
        params = [
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]
        params = encoder_params + params

        return params


class DRJModel_TextCnn(nn.Module):
    def __init__(self, out_size=300, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(DRJModel_TextCnn, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)
        # the below cls_layer is for multimodal baseline
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))
        # train_texts*train_imgs
        # train_y = self.head(torch.cat((train_texts, f), dim=1))
        train_y = self.head(f)
        return train_texts, train_imgs, train_y, instance_cls

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = list(map(id, self.encoder.parameters()))

        params = [
            {"params": self.encoder.parameters(), "lr": None},
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]

        return params


class DRJModel_Mgat(nn.Module):
    def __init__(self, input_size=768, out_size=300, num_label=2, freeze_id=-1):
        super(DRJModel_Mgat, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_label = num_label
        self.encoder = Encoder(input_size=self.input_size, out_size=self.out_size, freeze_id=freeze_id)
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size, out_features=self.out_size),
                                       nn.Tanh())
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)
        self.modal_score = nn.Sequential(nn.Linear(in_features=self.out_size, out_features=self.out_size), nn.Tanh(),
                                         nn.Linear(in_features=self.out_size, out_features=1))

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        # N, 1
        text_score = self.modal_score(train_texts)
        img_score = self.modal_score(train_imgs)
        # N, 2, 1
        score = F.softmax(torch.cat([text_score, img_score], dim=1)).unsqueeze(2)
        train_texts = self.cls_layer(train_texts)
        train_imgs = self.cls_layer(train_imgs)
        # N, 2, outsize
        train_text_img = torch.stack([train_texts, train_imgs], dim=1)
        # N, outsize
        y = self.head(torch.sum(score * train_text_img, dim=1).squeeze())
        return train_texts, train_imgs, y, instance_cls

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = self.encoder.get_parameters()
        params = [
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]
        params = encoder_params + params

        return params


class DRJModel_Mgat_TextCnn(nn.Module):
    def __init__(self, out_size=300, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(DRJModel_Mgat_TextCnn, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size, out_features=self.out_size),
                                       nn.Tanh())
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)
        self.modal_score = nn.Sequential(nn.Linear(in_features=self.out_size, out_features=self.out_size), nn.Tanh(),
                                         nn.Linear(in_features=self.out_size, out_features=1))

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        # N, 1
        text_score = self.modal_score(train_texts)
        img_score = self.modal_score(train_imgs)
        # N, 2, 1
        score = F.softmax(torch.cat([text_score, img_score], dim=1), dim=0).unsqueeze(2)
        train_texts = self.cls_layer(train_texts)
        train_imgs = self.cls_layer(train_imgs)
        # N, 2, outsize
        train_text_img = torch.stack([train_texts, train_imgs], dim=1)
        # N, outsize
        y = self.head(torch.sum(score * train_text_img, dim=1).squeeze())
        return train_texts, train_imgs, y, instance_cls


    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = self.encoder.get_parameters()
        params = [
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]
        params = encoder_params + params

        return params


class DRJModel_FusionMMD(nn.Module):
    def __init__(self, input_size=768, out_size=300, num_label=2, freeze_id=-1):
        super(DRJModel_FusionMMD, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_label = num_label
        self.encoder = Encoder(input_size=self.input_size, out_size=self.out_size, freeze_id=freeze_id)
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))
        # train_texts*train_imgs
        # train_y = self.head(torch.cat((train_texts, f), dim=1))
        train_y = self.head(f)
        return f, train_y, instance_cls

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = self.encoder.get_parameters()
        params = [
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]
        params = encoder_params + params

        return params


class DRJModel_FusionMMD_Textcnn(nn.Module):
    def __init__(self, out_size=300, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(DRJModel_FusionMMD_Textcnn, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)
        # the below cls_layer is for multimodal baseline
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))
        # train_texts*train_imgs
        # train_y = self.head(torch.cat((train_texts, f), dim=1))
        train_y = self.head(f)
        # for irm
        return f, train_y
        # for our model
        # return train_texts, train_imgs, train_y, f

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = list(map(id, self.encoder.parameters()))

        params = [
            {"params": self.encoder.parameters(), "lr": None},
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]

        return params

    def reset_weights(self, meta_weights):
        self.load_state_dict(deepcopy(meta_weights))


class DRJModel_FusionMMD(nn.Module):
    def __init__(self, input_size=768, out_size=300, num_label=2, freeze_id=-1):
        super(DRJModel_FusionMMD, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_label = num_label
        self.encoder = Encoder(input_size=self.input_size, out_size=self.out_size, freeze_id=freeze_id)
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))
        train_y = self.head(f)
        return f, train_y, instance_cls

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = self.encoder.get_parameters()
        params = [
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]
        params = encoder_params + params

        return params







class Generator(nn.Module):
    def __init__(self, out_size=300, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(Generator, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)
        # the below cls_layer is for multimodal baseline
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))
        return f

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = list(map(id, self.encoder.parameters()))
        params = self.encoder.get_parameters() + [{"params": filter(lambda p: id(p) not in encoder_params,
                                                                    self.parameters()), "lr": None}]

        return params


class Classifier(nn.Module):
    def __init__(self, out_size=300, num_label=2):
        super(Classifier, self).__init__()
        self.out_size = out_size
        self.num_label = num_label
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, f):
        train_y = self.head(f)
        return train_y


class DRJModel_TextCnn_Vis(nn.Module):
    def __init__(self, out_size=300, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(DRJModel_TextCnn_Vis, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)
        # the below cls_layer is for multimodal baseline
        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))
        # self.head = nn.Linear(in_features=(int(self.out_size / 2) +self.out_size), out_features=self.num_label)
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, train_texts, train_imgs):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))
        train_y = self.head(f)
        return train_texts, train_imgs, train_y, instance_cls, f

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = list(map(id, self.encoder.parameters()))

        params = [
            {"params": self.encoder.parameters(), "lr": None},
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]

        return params


class ResNet_Classifier(nn.Module):
    def __init__(self, input_size=768, out_size=300, num_label=2, droprate=0.2):
        super(ResNet_Classifier, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_label = num_label
        self.droprate = droprate
        self.visual_encoder = torchvision.models.resnet50(pretrained=True)
        self.visual_encoder = torch.nn.Sequential(*(list(self.visual_encoder.children())[:-1]))
        self.img_linear = nn.Sequential(nn.Linear(in_features=2048, out_features=1024), nn.ReLU(),
                                        # nn.Dropout(p=0.2),
                                        nn.Linear(in_features=1024, out_features=self.out_size),
                                        nn.LayerNorm(self.out_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.droprate)
                                        )

        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size, out_features=self.out_size), nn.ReLU())
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)

    def forward(self, imgs):
        # texts = self.bert_model(**texts)[0]
        with torch.no_grad():
            imgs = self.visual_encoder(imgs)
            imgs = imgs.squeeze()
        imgs = self.img_linear(imgs)
        f = self.cls_layer(imgs)
        train_y = self.head(f)
        return train_y


