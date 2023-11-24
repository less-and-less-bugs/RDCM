from torch.utils.data import Dataset
import json
import os
from torchvision import transforms
from PIL import Image


class PhemeSet(Dataset):
    def __init__(self, json_path="./phemeset.json", img_path="./pheme_dataset/images",
                 type=0, events=None, visual_type='resnet', stage='train'):
        """
        Args:
            json_path:
            type: Int. 0 is without dependency, 1 is with dependency.
            events: list. must not be None.
            visual_type='resnet' or 'vit'
            ---------------------------------- dataset statistics-----------------------------
            charliehebdo 923
            sydneysiege 419
            ferguson 351
            ottawashooting 256
            germanwings-crash 182
            'prince-toronto', 'ebola-essien', 'charliehebdo', 'putinmissing', 'gurlitt'
            ---------------------------------------------------------------------------------
        Returns:
            text
            img
            label

        """
        self.type = type
        self.events = events
        self.dataset = []
        self.img_prefix = img_path
        self.visual_type = visual_type
        with open(json_path, 'r') as f:
            pheme_set = json.load(f)
        self.dataset = []
        if self.events is None:
            exit()
        if isinstance(events, list):
            for event in self.events:
                [self.dataset.append(sample) for sample in pheme_set[event]]
        else:
            [self.dataset.append(sample) for sample in pheme_set[events]]
        self.stage = stage

        if self.visual_type == "resnet":
            self.tfms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        else:
            # vit
            self.tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

    def __getitem__(self, item):
        sample = self.dataset[item]
        sample: dict
        img = sample['image']
        img = os.path.join(self.img_prefix, img)
        # （3，224，224）
        img = Image.open(img).convert('RGB')
        img = self.tfms(img)
        # resize and transform
        label = sample['label']

        if self.type == 0:
            text = sample['text']
            return text, img, label
        elif self.type == 1:
            text = sample['dep']['token_cap']
            dep = sample['dep']['token_dep']
            return text, img, label, dep
        elif self.type == 2:
            text = sample['text']
            dep = sample['dep']['token_dep']
            return text, img, label, dep

    def __len__(self):
        return len(self.dataset)


class TwitterSet(Dataset):
    def __init__(self, json_path="./final_twitter.json", img_path="./twitter/images",
                 type=0, events=None, visual_type='resnet', stage='train'):
        """
        Args:
            json_path:
            img_path:
            type: Int. 0 is without dependency, 1 is with dependency.
            events: list. must not be None.
            visual_type='resnet' or 'vit'
            ---------------------------------- dataset statistics-----------------------------
            sandy:fake 5461,real 6841
            elephant:fake 13,real 0
            Passport:fake 44,real 2
            Livr:fake 9,real 0
            boston:fake 81,real 325 1
c           olumbianChemicals:fake 185,real 0
            PigFish:fake 14,real 0
            malaysia:fake 310,real 191 1
            bringback:fake 131,real 0
            underwater:fake 112,real 0
            sochi:fake 274,real 127 1
            ---------------------------------------------------------------------------------
        Returns:
            text
            img
            label

        """
        self.type = type
        self.events = events
        self.dataset = []
        self.img_prefix = img_path
        self.visual_type = visual_type
        with open(json_path, 'r') as f:
            pheme_set = json.load(f)
        self.dataset = []
        if self.events is None:
            exit()
        if isinstance(events, list):
            for event in self.events:
                [self.dataset.append(sample) for sample in pheme_set[event]]
        else:
            [self.dataset.append(sample) for sample in pheme_set[events]]
        self.stage = stage

        if self.visual_type == "resnet":
            # ResizeImage(256),
            # T.RandomResizedCrop(224)
            # # .Resize((224, 224))
            # if random_horizontal_flip:
            #     transforms.append(T.RandomHorizontalFlip())
            # if random_color_jitter:
            #     transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
            self.tfms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        else:
            # vit
            self.tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

    def __getitem__(self, item):
        sample = self.dataset[item]
        sample: list
        img = sample[2]
        img = os.path.join(self.img_prefix, img)
        # （3，224，224）
        img = Image.open(img).convert('RGB')
        img = self.tfms(img)
        # resize and transform
        label = sample[3]

        if self.type == 0:
            text = sample[1]
            return text, img, label

    def __len__(self):
        return len(self.dataset)