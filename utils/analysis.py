from torch.utils.data import DataLoader
import tqdm
import matplotlib
from sklearn.manifold import TSNE
import numpy as np

# do not display image in pycharm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as col
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from .meter import AverageMeter
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None):
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    # all_visual_features = []
    # all_textual_features = []
    with torch.no_grad():
        for i, (texts, images, target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            texts = texts.to(device)
            images = images.to(device)
            texts, images, target, instance_cls = feature_extractor(texts, images)
            texts = texts.cpu()
            images = images.cpu()
            target = torch.cat([texts, images], dim=1)
            # target = images

            # all_visual_features.append(images)
            # all_textual_features.append(texts)
            all_features.append(target)
    return torch.cat(all_features, dim=0)
    #
    # return torch.cat(all_visual_features, dim=0), torch.cat(all_textual_features, dim=0)


def cal_kl(train_target_loader, test_loader, feature_extractor, device, max_iter=100):
    with torch.no_grad():
        kl_divergence = 0
        len_test = len(test_loader)
        len_source = len(train_target_loader)
        for i, (texts, images, target) in enumerate(tqdm.tqdm(train_target_loader)):
            if i > max_iter:
                return kl_divergence/(max_iter*len_test)
            texts = texts.to(device)
            images = images.to(device)
            texts, images, target, instance_cls = feature_extractor(texts, images)
            target_source = torch.cat([texts, images], dim=1)
            target_source = F.softmax(target_source, dim=1)

            for j, (texts, images, target) in enumerate(test_loader):
                texts = texts.to(device)
                images = images.to(device)
                texts, images, target, instance_cls = feature_extractor(texts, images)
                target_test = torch.cat([texts, images], dim=1)
                target_test = F.softmax(target_test, dim=1)

                kl_divergence = kl_divergence + F.kl_div(target_source.log(), target_test, reduction="mean")
        return kl_divergence/(len_source*len_test)


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.
    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier
    Returns:
        :math:`\mathcal{A}`-distance
    """
    dataset_lan = np.min(np.array([500, target_feature.shape[0], source_feature.shape[0]]))
    print(dataset_lan)
    source_label = torch.ones((dataset_lan, 1))
    target_label = torch.zeros((dataset_lan, 1))

    sample_indices_source = np.random.choice(source_feature.shape[0], size=dataset_lan, replace=False)
    sample_indices_target = np.random.choice(target_feature.shape[0], size=dataset_lan, replace=False)
    feature = torch.cat([source_feature[sample_indices_source], target_feature[sample_indices_target]], dim=0).numpy()
    label = torch.cat([source_label, target_label], dim=0).numpy()
    a_distance = 2.0
    acc_avg = 0
    for epoch in range(3):
        print("--------------------run----------------------")
        X_train, X_val, Y_train, Y_val = train_test_split(feature, label, test_size=0.3, random_state=np.random.randint(0, 10000, 1).item())
        print("the length of X_train: {} and the length of Y_train: {}".format(len(X_train), len(X_val)))
        clf = svm.NuSVC(gamma="auto")
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(Y_val, y_pred)
        print("-------------------{}----------------".format(acc))
        acc_avg = acc + acc_avg
    error = 1 - acc_avg / 3
    a_distance = 2 * (1 - 2 * error)
    print(" A-dist: {}".format(a_distance))

    return a_distance

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


