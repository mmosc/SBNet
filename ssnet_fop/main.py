
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
from sklearn import preprocessing
import torch.nn as nn

from tqdm import tqdm
from binary_classification_model import SingleBranchWithDownproject, SingleBranchWithPadding

import online_evaluation


def read_feature(file_path, ids):
    features = pd.read_csv(file_path, sep='\t')
    features = features.set_index('id')

    # features of the list of first tracks
    features_i = features.loc[ids]
    features_i = np.asarray(features_i)
    return features_i


def read_data(split):
    """
    Reads, processes, and returns the features and the labels.
    Labels:
     - read the csv file with the id pairs and the labels
     - process the labels
    Features:
     - read the csv file with the features
     - for the list of first tracks (i), filter the features
     - for the list of second tracks (j), filter the features
     - convert both to numpy arrays
    This function is used to load both the training and validation/test sets
    :param FLAGS: additional parameters
    :param split: whether to read the train, val, or test split
    :return: features_i, features_j, train_label. np.arrays containing the input features of the first and second tracks (i, j)
    and of the labels
    """
    fi_name, fj_name = FLAGS.feature_i, FLAGS.feature_j
    print(f'Modalities: {fi_name}\t {fj_name}')
    labels_file = f'../data/binary_classification/binary_{split}.tsv'

    print(f'Reading {fi_name}, {fj_name} Train')
    train_data = pd.read_csv(labels_file, sep='\t')
    # columns i and j are the ids
    i_ids = train_data['i'].tolist()
    j_ids = train_data['j'].tolist()

    # column is_match is the label
    train_label = train_data['is_match']
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)

    # features
    train_file_i = f'../data/binary_classification/id_{fi_name}_mmsr.tsv'
    features_i = read_feature(train_file_i, i_ids)


    # features
    train_file_j = f'../data/binary_classification/id_{fj_name}_mmsr.tsv'
    features_j = read_feature(train_file_j, j_ids)

    return features_i, features_j, train_label


def get_batch(batch_index, batch_size, labels, i_f_lst, j_f_lst):
    """
    takes the full dataset and returns  features of first, features of second, and labels; all filtered for the given batch
    :param batch_index: the index of the batch
    :param batch_size: the size of the batch
    :param labels: the labels of the full dataset
    :param i_f_lst: the features of the first tracks, for the full dataset
    :param j_f_lst: the features of the second tracks, for the full dataset
    :return: batch_feat_i
    :return: batch_feat_j
    :return: batch_labels
    """
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size

    batch_feat_i = np.asarray(i_f_lst[start_ind:end_ind])
    batch_feat_j = np.asarray(j_f_lst[start_ind:end_ind])
    batch_labels = np.asarray(labels[start_ind:end_ind])

    return batch_feat_i, batch_feat_j, batch_labels

def init_weights(m):
    """
    Initialize the weights of the linear layers of the model
    :param m: pytorch model
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(i_train_data, j_train_data, train_label, i_test_data, j_test_data, test_label):
    """
    The actual training.
    It also stores:
     - the best model over epochs
     - the values of the loss and of the validation metrics over epochs

    Training set:
    :param i_train_data: np.array storing the features of the first trakcs
    :param j_train_data: np.array storing the features of the second trakcs
    :param train_label: labels for each pair

    Test set (used for early stopping, so actually it shoyld be called validation set)
    :param i_test_data: np.array storing the features of the first trakcs
    :param j_test_data: np.array storing the features of the second trakcs
    :param test_label: labels for each pair
    :return loss_plot: values of the loss over epochs
    :return min_eer: minimum error over epochs
    :return max_auc: maximum auc over epochs
    """

    # initialize the model
    if FLAGS.merging_technique == 'downprojection':
        model = SingleBranchWithDownproject(FLAGS, i_train_data.shape[1], j_train_data.shape[1], DEVICE)
    elif FLAGS.merging_technique == 'padding' or i_train_data.shape[1] == j_train_data.shape[1]:
        model = SingleBranchWithPadding(FLAGS, i_train_data.shape[1], j_train_data.shape[1], DEVICE)
    else:
        print(f'Merging technique {FLAGS.merging_technique} not recognized!')

    model.apply(init_weights)

    # set loss to binary cross entropy from logits (softmax "included" in the loss)
    # bce_loss = nn.BCEWithLogitsLoss().cuda()
    bce_loss = nn.BCELoss().cuda()

    model.to(DEVICE)
    bce_loss.to(DEVICE)

    if DEVICE == 'cuda':
        cudnn.benchmark = True
    
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Model type {model.name}  + Number of params: {n_parameters}')
    
    # for alpha in FLAGS.alpha_list:
    eer_list = []
    epoch = 1
    num_of_batches = (len(train_label) // FLAGS.batch_size)
    loss_plot = []
    auc_list = []
    loss_per_epoch = 0
    txt_dir = 'output'
    fi_name, fj_name = FLAGS.feature_i, FLAGS.feature_j
    save_dir = f'fc2_{fi_name}_{fj_name}_{FLAGS.merging_technique}_{FLAGS.save_dir}'
    txt = '%s/binary_classification_log_%03d_f1_%s_f2_%s_%s.txt'%(txt_dir, FLAGS.max_num_epoch, FLAGS.feature_i, FLAGS.feature_j, FLAGS.merging_technique)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    with open(txt, 'w+') as f:
        f.write('EPOCH\tLOSS\tEER\tAUC\tS_FAC\tD_FAC\n')

    save_best = 'best_%s'%(save_dir)

    if not os.path.exists(save_best):
        os.makedirs(save_best)
    with open(txt, 'a+') as f:
        while (epoch < FLAGS.max_num_epoch):
            print(f'{fi_name}_{fj_name}\tEpoch {epoch}')
            for idx in tqdm(range(num_of_batches)):
                i_train_batch, j_train_batch, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, i_train_data, j_train_data)
                # voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
                loss_tmp = train(
                    i_train_batch,
                    j_train_batch,
                    batch_labels,
                    model, optimizer, bce_loss)

                loss_per_epoch += loss_tmp

            loss_per_epoch /= num_of_batches

            loss_plot.append(loss_per_epoch)
            eer, auc = online_evaluation.test(FLAGS, model, i_test_data, j_test_data, test_label, DEVICE)


            
            eer_list.append(eer)
            auc_list.append(auc)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d_%0.3f.pth.tar'%(epoch, eer*100))

            print('==> Epoch: %d/%d Loss: %0.4f, Min_EER: %0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch, min(eer_list)))

            if eer <= min(eer_list):
                min_eer = eer
                max_auc = auc
                save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict()}, save_best, 'checkpoint.pth.tar')

            f.write('%04d\t%0.4f\t%0.2f\t%0.2f\n'%(epoch, loss_per_epoch, eer, auc))
            loss_per_epoch = 0
            epoch += 1

                
        return loss_plot, min_eer, max_auc

def train(i_train_batch, j_train_batch, labels, model, optimizer, bce_loss):
    """
    Training function for a single batch.

    :param i_train_batch: features for first tracks (i) in the batch
    :param j_train_batch: features for second tracks (j) in the batch
    :param labels: labels of the batch
    :param model: pytorch model to optimize
    :param optimizer: optimizer
    :param bce_loss: binary cross entropy loss
    :return: loss of the batch
    """
    
    average_loss = RunningAverage()

    model.train()
    i_train_batch = torch.from_numpy(i_train_batch).float()
    j_train_batch = torch.from_numpy(j_train_batch).float()
    labels = torch.from_numpy(labels)
    
    i_train_batch, j_train_batch, labels = i_train_batch.to(DEVICE), i_train_batch.to(DEVICE), labels.to(DEVICE)

    i_train_batch, j_train_batch, labels = Variable(i_train_batch), Variable(j_train_batch), Variable(labels)
    confidence = model.train_forward(i_train_batch, j_train_batch)

    loss = bce_loss(confidence, labels.float())

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())

    optimizer.step()

    return average_loss.avg()

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average
 
def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed. Default 1')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training. Default is cuda, if cuda is not available, uses cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--save_dir', type=str, default='model', help='Directory for saving checkpoints. Default model')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate. Default: 1e-5')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training. Default 128')
    parser.add_argument('--max_num_epoch', type=int, default=500, help='Max number of epochs to train, number. Default 500')
    parser.add_argument('--intermediate_emb', type=int, default=256, help='Intermediate Layer. Default 256')
    parser.add_argument('--dim_embed', type=int, default=128, help='Embedding Size. Default 128')
    parser.add_argument('--feature_i', type=str, default='mfcc_bow', help='feature i (first modality). Default mfcc_bow')
    parser.add_argument('--feature_j', type=str, default='mfcc_bow', help='feature j (second modality). Default mfcc_bow')
    parser.add_argument('--merging_technique', type=str, default='downproject', help='whether to downproject or pad if there is a dimension mismatch. Default downproject')

    global FLAGS, DEVICE

    FLAGS, unparsed = parser.parse_known_args()
    # DEVICE = 'cpu'

    torch.manual_seed(FLAGS.seed)

    if torch.cuda.is_available() and FLAGS.device == 'cuda':
        DEVICE = 'cuda'
        torch.cuda.manual_seed(FLAGS.seed)
        print('Running on CUDA')
    else:
        DEVICE = 'cpu'
        print('Running on CPU')

    i_train_data, j_train_data, train_label = read_data('train')
    i_test_data, j_test_data, test_label = read_data('test')
    
    main(i_train_data, j_train_data, train_label, i_test_data, j_test_data, test_label)
