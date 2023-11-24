import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import argparse
import torch.optim as optim
from utils import CompleteLogger, seed_everything, collect_feature, calculate, visualize
from utils import ProgressMeter, AverageMeter, get_accuracy, compute_mean_std,get_three_source_loader
from model import JointMultipleKernelMaximumMeanDiscrepancy, GaussianKernel, ContrastiveLoss_coot, Intro_alignment_loss
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
from model import DRJModel_TextCnn, DRJModel_Mgat_TextCnn
from comet_ml import Experiment
import numpy as np
from sklearn import metrics
import random
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
This file is for uni-modal textcnn using spacy, roberta, bert vocabulary
"""

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')


def get_parser():
    parser = argparse.ArgumentParser(description='Multimodal TextCnn')
    # dataset parameters and log parameter
    parser.add_argument('-d', '--data', metavar='DATA', default='Pheme', choices=['Pheme', 'Twitter', 'Cross'],
                        help='dataset: Pheme, Fakeddit, Twitter')
    parser.add_argument('-s', '--source', help='source domain(s)', type=str,
                        default='charliehebdo-sydneysiege-ottawashooting')
    parser.add_argument('-t', '--target', help='target domain(s)', default='ferguson')
    parser.add_argument("--tokenizer_type", type=str, default='roberta',
                        help="the tokenizer of text consisting of bert, roberta, spacy")
    parser.add_argument("--textcnn_mode", type=str, default="roberta-non",
                        choices=["rand", "roberta-yes", "roberta-non",
                                 "bert-yes", "bert-non"],
                        help="The embedding mode of textcnn")
    parser.add_argument("--tag", type=str, default="roberta-non",
                        help="the tags for comet")
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--patience', default=5, type=int, metavar='M',
                        help='patience')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--seed', default=0
                        , type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cls_dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--max_iter', default=30, type=int,
                        help='the maximum number of iteration in each epoch ')
    parser.add_argument('--gat', default="false", type=str,
                        help='Whether to use modality gat mechanism')
    parser.add_argument('--freeze_resnet', default=8, type=int,
                        help='finetune layers of resnet')
    parser.add_argument('--d_rop', default=0.3, type=float,
                        help='d_rop rate of neural network model')
    # inter-domain parameter
    parser.add_argument('--linear', default='false', type=str,
                        help='whether use the linear version of MMD')
    parser.add_argument('--da', default='false', type=str,
                        help='finetune layers of resnet')
    parser.add_argument('--lambda1', default=0., type=str,
                        help='the trade-off hyper-parameter for inter-domain transfer loss. 0 means non-transfer')
    parser.add_argument('--tsigma', default="1#2#3#4", type=str,
                        help='the sigma of Gaussian Kernel for textual feature')
    parser.add_argument('--vsigma', default="1#2#3#4", type=str,
                        help='the sigma of Gaussian Kernel for visual feature')
    parser.add_argument('--ysigma', default="1#2#3#4", type=str,
                        help='the sigma of Gaussian Kernel for visual feature')
    parser.add_argument('--yadapt', default='false', type=str, help='whether use adversarial theta')
    # intra-domain parameter
    parser.add_argument('--intratheta', default='true', type=str, help='whether use the mlp for intra loss')
    parser.add_argument('--lambda2', default=1., type=str,
                        help='the trade-off hyper-parameter for intra-domain transfer loss')
    parser.add_argument('--temperature', default=3., type=float,
                        help='the temperature for contrastive loss')
    parser.add_argument('--threshold', default=3., type=float,
                        help='the threshold for contrastive loss')
    parser.add_argument('--ctsize', default=64, type=int,
                        help='the size for contrastive learning')
    # log
    parser.add_argument("--log", type=str, default='jan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model.")

    args = parser.parse_args()

    if args.gat in ["True", "true"]:
        args.gat = True
    else:
        args.gat = False
    #
    if args.linear in ['true', 'True']:
        args.linear = True
    else:
        args.linear = False
    if args.da in ['true', 'True']:
        args.da = True
    else:
        args.da = False

    if args.yadapt in ['true', 'True']:
        args.yadapt = True
    else:
        args.yadapt = False

    if args.intratheta in ['true', 'True']:
        args.intratheta = True
    else:
        args.intratheta = False
    args.lambda1 = float(args.lambda1)
    args.lambda2 = float(args.lambda2)

    args.tsigma = [float(i) for i in str(args.tsigma).strip().split('#')]
    args.vsigma = [float(i) for i in str(args.vsigma).strip().split('#')]
    args.ysigma = [float(i) for i in str(args.ysigma).strip().split('#')]


    return args


def main(args, experiment=None):
    logger = CompleteLogger(args.log, args.phase)
    multi_source_domains_loader, multi_source_domains_iter, train_target_loader, train_target_iter, test_loader = get_three_source_loader(
        args=args)

    print(args)

    if args.gat:
        classifier = DRJModel_Mgat_TextCnn(out_size=args.cls_dim, num_label=2, freeze_id=args.freeze_resnet,
                                           d_prob=args.d_rop, kernel_sizes=[3, 4, 5], num_filters=100,
                                           mode=args.textcnn_mode, dataset_name=args.data)
    else:
        classifier = DRJModel_TextCnn(
            out_size=args.cls_dim, num_label=2, freeze_id=args.freeze_resnet,
            d_prob=args.d_rop, kernel_sizes=[3, 4, 5], num_filters=100,
            mode=args.textcnn_mode, dataset_name=args.data)

    jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
        kernels=(
            [GaussianKernel(sigma=k, track_running_stats=False) for k in args.tsigma],
            [GaussianKernel(sigma=k, track_running_stats=False) for k in args.vsigma]
        ),
        linear=args.linear,
    )

    # jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
    #     kernels=(
    #         [GaussianKernel(alpha=2 ** k) for k in args.tsigma],
    #         [GaussianKernel(alpha=2 ** k) for k in args.vsigma]
    #     ),
    #     linear=args.linear,
    # )


    intra_loss = Intro_alignment_loss(theta=args.intratheta, temperature=args.temperature, threshold=args.threshold,
                                      input_dim=args.cls_dim,
                                      output_dim=args.cls_dim)
    classifier.to(device)
    jmmd_loss.to(device)
    intra_loss.to(device)

    parameters = classifier.get_parameters()
    for para in parameters:
        para["lr"] = args.lr
    parameters += [{"params": jmmd_loss.parameters(), 'lr': args.lr}]
    parameters += [{"params": intra_loss.parameters(), 'lr': args.lr}]
    # define optimizer and lr scheduler
    # optimizer = SGD(params=parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    # lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + 0.001 * float(x)) ** (-0.75))
    #
    optimizer = optim.Adam(params=parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=args.wd,
                           amsgrad=True)
    # optimizer_jmmd = optim.Adam(params=jmmd_loss.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    #                        weight_decay=5e-4,
    #                        amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, verbose=True)
    # lr_scheduler_jmmd = ReduceLROnPlateau(optimizer_jmmd, mode='max', factor=0.1, patience=args.patience, verbose=True)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        classifier.head = nn.Identity()
        feature_extractor = classifier
        # source_visual_feature, source_textual_feature = collect_feature(train_source_loader, feature_extractor, device)
        # target_visual_feature, target_textual_feature = collect_feature(train_target_loader, feature_extractor, device)
        # need update
        source_feature = collect_feature(multi_source_domains_loader, feature_extractor, device)
        target_feature = collect_feature(multi_source_domains_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = os.path.join(logger.visualize_directory, 'TSNE.pdf')
        visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        # acc1 = validate(test_loader, classifier, device)
        losses, top1, f1_scores, recall_scores, precession_scores, results = validate_print(test_loader, classifier, device)
        torch.save(results, os.path.join(args.log, 'visr.pt'))
        return 0, 0

    # start training
    best_acc1 = 0.
    acc1_store = []
    f1_store = []
    prefix = str(args.seed) + "-" + args.target
    for epoch in range(args.epochs):
        # train for one epoch
        # evaluate on validation set
        cls_losses_source, cls_source, f1_source, domain_dis, intra_dis, all_loss  = train_transfer(train_source_iter=multi_source_domains_iter,
                                                                              train_target_iter=train_target_iter,
                                                                              model=classifier,
                                                                              optimizer=optimizer,
                                                                              epoch=epoch, jmmd_loss=jmmd_loss,
                                                                              intra_loss_con=intra_loss,
                                                                              args=args)

        cls_loss_target, acc1, f1_target, recall_target, pre_target = validate(test_loader, classifier, device)
        target_metircs = {prefix + "-target_acc": acc1, prefix + "-target_loss": cls_loss_target,
                          prefix + "-target_recall": recall_target, prefix + "-target_pre": pre_target}
        source_metircs = {prefix + "-source_acc": cls_source, prefix + "-source_cls_loss": cls_losses_source,
                          prefix+"-source_inter_loss": domain_dis,  prefix+"-source_intra_loss": intra_dis,
                          prefix+"-source_loss": all_loss}
        experiment.log_metrics(target_metircs, epoch=epoch)
        experiment.log_metrics(source_metircs, epoch=epoch)

        f1_store.append(f1_target)
        acc1_store.append(acc1)
        lr_scheduler.step(float(acc1))
        # lr_scheduler_jmmd.step(float(acc1))
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    acc1_store.sort()
    f1_store.sort()
    three_avg_acc = sum(acc1_store[-3:]) / 3
    three_avg_f1 = sum(f1_store[-3:]) / 3
    print("best_acc1 = {:.4f}".format(three_avg_acc))

    logger.close()
    return three_avg_acc, three_avg_f1


def train_transfer(train_source_iter, train_target_iter, model, optimizer, epoch, jmmd_loss, intra_loss_con, args):
    """
    set lambda=0 for baseline removing inter-modality adaptation
    Args:
        train_source_iter:
        model:
        optimizer:
        epoch:
    Returns:

    """
    batch_time = AverageMeter('Time', ':4.2f')
    losses = AverageMeter('Loss', ':3.4f')
    cls_losses = AverageMeter('CLS Loss', ':3.4f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    intra_losses = AverageMeter('Intra Loss', ':3.4f')
    domain_dis = AverageMeter('Domain Discrepancy', ':3.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.4f')
    f1_scores = AverageMeter('F1_score', ':3.4f')
    # max_iters = len(train_source_iter)
    progress = ProgressMeter(
        args.max_iter,
        [batch_time, losses, cls_losses, trans_losses, intra_losses, domain_dis, cls_accs, f1_scores],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    jmmd_loss.train()
    intra_loss_con.train()

    end = time.time()

    # lambda1 = lambda1*10
    # train_source_iter = args.max_iter
    real_label_list = []
    predicted_y_list = []
    lambda1 = 1
    for batch_idx in range(args.max_iter):
        # source
        texts_s1, imgs_s1, labels_s1 = next(train_source_iter[0])
        texts_s2, imgs_s2, labels_s2 = next(train_source_iter[1])
        texts_s3, imgs_s3, labels_s3 = next(train_source_iter[2])
        texts_t, imgs_t, labels_t = next(train_target_iter)

        labels_s1 = labels_s1.to(device)
        texts_s1 = texts_s1.to(device)
        imgs_s1 = imgs_s1.to(device)

        labels_s2 = labels_s2.to(device)
        texts_s2 = texts_s2.to(device)
        imgs_s2 = imgs_s2.to(device)

        labels_s3 = labels_s3.to(device)
        texts_s3 = texts_s3.to(device)
        imgs_s3 = imgs_s3.to(device)

        # labels_t = labels_t.to(device)
        texts_t = texts_t.to(device)
        imgs_t = imgs_t.to(device)

        ft_s1, fv_s1, y_s1, instance_s1 = model(train_texts=texts_s1, train_imgs=imgs_s1)
        ft_s2, fv_s2, y_s2, instance_s2 = model(train_texts=texts_s2, train_imgs=imgs_s2)
        ft_s3, fv_s3, y_s3, instance_s3 = model(train_texts=texts_s3, train_imgs=imgs_s3)
        ft_t, fv_t, y_t, instance_t = model(train_texts=texts_t, train_imgs=imgs_t)

        ft_s = torch.cat([ft_s1, ft_s2, ft_s3], dim=0)
        fv_s = torch.cat([fv_s1, fv_s2, fv_s3], dim=0)
        labels_s = torch.cat([labels_s1, labels_s2, labels_s3], dim=0)
        instances = torch.cat([instance_s1, instance_s2, instance_s3], dim=0)

        ctindex = random.sample(list(np.arange(ft_s.size(0))), args.ctsize)

        if args.lambda2 == 0:
            intra_loss = torch.zeros(1).cuda()
        else:
            intra_loss, num_intra = intra_loss_con(ft_s[ctindex], fv_s[ctindex], labels_s[ctindex], instances[ctindex])

        if args.lambda1 == 0:
            transfer_loss = torch.zeros(1).cuda()
        else:
            if args.da:
                if args.yadapt:
                    y_s1 = F.softmax(y_s1, dim=1)
                    y_s2 = F.softmax(y_s2, dim=1)
                    y_s3 = F.softmax(y_s3, dim=1)
                    y_t = F.softmax(y_t, dim=1)
                    transfer_loss = (jmmd_loss((ft_s1, fv_s1, y_s1), (ft_t, fv_t, y_t)) + jmmd_loss((ft_s2, fv_s2, y_s2),
                    (ft_t, fv_t, y_t)) + jmmd_loss((ft_s3, fv_s3, y_s3), (ft_t, fv_t, y_t)) +
                                     jmmd_loss((ft_s1, fv_s1, y_s1), (ft_s2, fv_s2, y_s2)) + jmmd_loss((ft_s1, fv_s1, y_s1),
                                    (ft_s3, fv_s3, y_s3)) +
                                     jmmd_loss((ft_s2, fv_s2, y_s2), (ft_s3, fv_s3, y_s3))) / 6
                else:
                    transfer_loss = (jmmd_loss((ft_s1, fv_s1), (ft_t, fv_t)) + jmmd_loss((ft_s2, fv_s2),
                                                                                         (ft_t, fv_t)) + jmmd_loss(
                        (ft_s3,
                         fv_s3), (ft_t, fv_t)))/3 + (jmmd_loss((ft_s1, fv_s1), (ft_s2, fv_s2)) + jmmd_loss((ft_s1, fv_s1),
                                                                                                       (ft_s3, fv_s3)) +
                                     jmmd_loss((ft_s2, fv_s2), (ft_s3, fv_s3))) / 3
            else:
                if args.yadapt:
                    y_s1 = F.softmax(y_s1, dim=1)
                    y_s2 = F.softmax(y_s2, dim=1)
                    y_s3 = F.softmax(y_s3, dim=1)
                    transfer_loss = (jmmd_loss((ft_s1, fv_s1, y_s1), (ft_s2, fv_s2, y_s2)) +
                                     jmmd_loss((ft_s1, fv_s1, y_s1), (ft_s3, fv_s3, y_s3)) +
                                 jmmd_loss((ft_s2, fv_s2, y_s2), (ft_s3, fv_s3, y_s3))) / 3
                else:
                    transfer_loss = (jmmd_loss((ft_s1, fv_s1), (ft_s2, fv_s2)) + jmmd_loss((ft_s1, fv_s1), (ft_s3, fv_s3)) +
                                 jmmd_loss((ft_s2, fv_s2), (ft_s3, fv_s3))) / 3

        domain_dis.update(transfer_loss.item(), 1)
        # transfer_loss = transfer_loss * lambda1 / 6 * 5
        y_s = torch.cat([y_s1, y_s2, y_s3], dim=0)
        cls_loss = F.cross_entropy(y_s,  labels_s)

        trans_losses.update(transfer_loss.item(), 1)
        intra_losses.update(intra_loss.item(), 1)
        cls_losses.update(cls_loss.item(), 1)

        loss = cls_loss + transfer_loss * lambda1 * args.lambda1 + intra_loss * args.lambda2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cls_acc = get_accuracy(labels_s.detach().clone(), y_s.detach().clone())
        # the number of samples predicted correctly
        real_label_list = real_label_list + labels_s.cpu().detach().clone().numpy().tolist()
        predicted_y_list = predicted_y_list + (y_s[:, 0] < y_s[:, 1]).cpu().long().numpy().tolist()
        losses.update(loss.item(), 1)

        cls_accs.update(cls_acc.item(), y_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # if batch_idx == max_iters - 1:
    f1 = metrics.f1_score(np.array(real_label_list), np.array(predicted_y_list))
    f1_scores.update(float(f1), 1)
    progress.display(len(train_source_iter))

    return cls_losses.avg, cls_accs.avg, f1_scores.avg, domain_dis.avg, intra_losses.avg, losses.avg


def validate(val_loader, model, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc', ':6.4f')
    f1_scores = AverageMeter('F1_score', ':3.4f')
    recall_scores = AverageMeter('Recall_score', ':3.4f')
    precession_scores = AverageMeter('Precession_score', ':3.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, f1_scores, recall_scores, precession_scores], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    real_label_list = []
    predicted_y_list = []
    with torch.no_grad():
        end = time.time()
        for i, (texts, imgs, target) in enumerate(val_loader):
            texts = texts.to(device)
            images = imgs.to(device)
            target = target.to(device)

            # compute output
            train_texts, train_imgs, output, instance_cls = model(train_texts=texts, train_imgs=images)
            # output = model(images)
            loss = F.cross_entropy(output, target)
            real_label_list = real_label_list + target.detach().cpu().clone().numpy().tolist()
            predicted_y_list = predicted_y_list + (output[:, 0] < output[:, 1]).cpu().long().numpy().tolist()
            acc1 = get_accuracy(target.detach().clone(), output.detach().clone())
            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        f1 = metrics.f1_score(np.array(real_label_list), np.array(predicted_y_list))
        recall_score = metrics.recall_score(np.array(real_label_list), np.array(predicted_y_list))
        precession_score = metrics.precision_score(np.array(real_label_list), np.array(predicted_y_list))
        f1_scores.update(float(f1), 1)
        recall_scores.update(float(recall_score), 1)
        precession_scores.update(float(precession_score), 1)
        # tn, fp, fn, tp
        progress.display(-1)
        # if i % args.print_freq == 0:
        #     progress.display(i)

        print('Average Acc {top1.avg: .4f}'.format(top1=top1))

    return losses.avg, top1.avg, f1_scores.avg, recall_scores.avg, precession_scores.avg


def validate_print(val_loader, model, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc', ':6.4f')
    f1_scores = AverageMeter('F1_score', ':3.4f')
    recall_scores = AverageMeter('Recall_score', ':3.4f')
    precession_scores = AverageMeter('Precession_score', ':3.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, f1_scores, recall_scores, precession_scores], prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    real_label_list = []
    predicted_y_list = []
    with torch.no_grad():
        end = time.time()
        for i, (texts, imgs, target) in enumerate(val_loader):
            texts = texts.to(device)
            images = imgs.to(device)
            target = target.to(device)

            # compute output
            train_texts, train_imgs, output, instance_cls = model(train_texts=texts, train_imgs=images)
            # output = model(images)
            loss = F.cross_entropy(output, target)
            real_label_list = real_label_list + target.detach().cpu().clone().numpy().tolist()
            predicted_y_list = predicted_y_list + (output[:, 0] < output[:, 1]).cpu().long().numpy().tolist()
            acc1 = get_accuracy(target.detach().clone(), output.detach().clone())
            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        f1 = metrics.f1_score(np.array(real_label_list), np.array(predicted_y_list))
        recall_score = metrics.recall_score(np.array(real_label_list), np.array(predicted_y_list))
        precession_score = metrics.precision_score(np.array(real_label_list), np.array(predicted_y_list))
        f1_scores.update(float(f1), 1)
        recall_scores.update(float(recall_score), 1)
        precession_scores.update(float(precession_score), 1)
        # tn, fp, fn, tp
        progress.display(-1)
        # if i % args.print_freq == 0:
        #     progress.display(i)

        print('Average Acc {top1.avg: .4f}'.format(top1=top1))

    return losses.avg, top1.avg, f1_scores.avg, recall_scores.avg, precession_scores.avg, [real_label_list, predicted_y_list]


if __name__ == '__main__':
    # must record every result
    args = get_parser()
    seed = [0, 42, 1024]
    if args.data == "Pheme":
        source = ["charliehebdo", "sydneysiege", "ottawashooting", "ferguson"]
        log_dir = ["sofc", "cofs", "csfo", "csof"]
    elif args.data == 'Cross':
        source = ["malaysia",  "sandy", "sydneysiege", "ottawashooting"]
        source_list = [["charliehebdo","ottawashooting", "ferguson"], ["charliehebdo","ottawashooting", "ferguson"], ["sandy", "boston", "sochi"], ["sandy", "boston", "sochi"]]
        log_dir = ["cofm", "cofa", "abis", "abio"]
    else:
        source = ["sandy", "boston", "malaysia", "sochi"]
        log_dir = ["bmia", "amib", "abim", "abmi"]
    # experiment = None

    experiment = Experiment(
        api_key="",
        project_name="",
        workspace="",
    )
    experiment.set_name(str(args.epochs) + str(args.da) + '-' + str(args.lambda1) + str(args.lambda2) +
                        '-' + str(args.tsigma) + '-' + str(args.tsigma))
    experiment.add_tag(args.tag)

    hyper_params = {
        "lr": args.lr,
        "epoch": args.epochs,
        "dataset": args.data,
        "da": args.da,

        "lambda1": args.lambda1,
        "vsigma": args.vsigma,
        "tsigma": args.tsigma,
        # "ysigma": args.ysigma,

        "lambda2": args.lambda2,

    }
    experiment.log_parameters(hyper_params)
    acc_seeds = []
    f1_seeds = []
    ori_log = args.log
    for seed_random in seed:
        args.seed = seed_random
        seed_everything(seed=args.seed)
        acc_seed = []
        f1_seed = []
        for i, target_domain in enumerate(source):
            index_list = [0, 1, 2, 3]
            index_list.remove(i)
            args.target = target_domain
            if args.data == 'Cross':
                args.source = source_list[i]
            else:
                args.source = [source[j_index] for j_index in index_list]
            args.log = "-".join([ori_log, str(args.seed)]) + '/' + log_dir[i]
            acc, f1 = main(args, experiment)
            acc_seed.append(acc)
            f1_seed.append(f1)


        acc_seeds.append(acc_seed)
        f1_seeds.append(f1_seed)
    mean_value_acc, std_value_acc, final_mean_acc, final_std_acc = compute_mean_std(acc_seeds)
    mean_value_f1, std_value_f1, final_mean_f1, final_std_f1 = compute_mean_std(f1_seeds)
    event_metric = {}
    for i, domain in enumerate(source):
        event_metric[domain] = [mean_value_acc[i], std_value_acc[i], mean_value_f1[i], std_value_f1[i]]
    # print(event_metric)
    final_metric_para = {"final_acc": final_mean_acc, "final_acc_std": final_std_acc,
                    "final_f1": final_mean_f1, "final_f1_std": final_std_f1}
    final_mean_acc_event = {}
    for i, domain in enumerate(source):
        final_mean_acc_event[domain] = float(event_metric[domain][0])
    experiment.log_parameters(event_metric)
    experiment.log_parameters(final_metric_para)
