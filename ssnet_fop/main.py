
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
from retrieval_model import FOP

import online_evaluation


def read_data(split, FLAGS):
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
    :param FLAGS:
    :return:
    """
    
    print('Split Type: %s'%(FLAGS.split_type))
    labels_file = f'../data/binary_{split}.csv'
    # for now let's focus on one feature only
    if FLAGS.split_type == 'mfcc_only':
        print('Reading MFCC Train')
        train_data = pd.read_csv(labels_file)
        # columns i and j are the ids
        i_ids = train_data['i'].tolist()
        j_ids = train_data['j'].tolist()

        # column is_match is the label
        train_label = train_data['is_match']
        le = preprocessing.LabelEncoder()
        le.fit(train_label)
        train_label = le.transform(train_label)


        # features
        train_file_mfccs = '/opt/datasets/Music4All/music4all/multimodal_full/id_mfcc_bow.csv'
        features = pd.read_csv(train_file_mfccs)
        features = features.set_index('ID')

        # features of the list of first tracks
        features_i = features.loc[i_ids]
        features_i = np.asarray(features_i)

        # features of the list of first tracks
        features_j = features.loc[j_ids]
        features_j = np.asarray(features_j)

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
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(i_train_data, j_train_data, train_label, i_test_data, j_test_data, test_label):
    """
    The actual training.

    :param i_train_data: np.array storing the features of the first trakcs
    :param j_train_data: np.array storing the features of the second trakcs
    :param train_label: labels for each pair
    :return:
    """

    # initialize the model
    model = FOP(FLAGS, i_train_data.shape[1], j_train_data.shape[1])
    model.apply(init_weights)

    # set loss to binary cross entropy from logits (softmax "included" in the loss)
    # bce_loss = nn.BCEWithLogitsLoss().cuda()
    bce_loss = nn.BCELoss().cuda()

    if FLAGS.cuda:
        model.cuda()
        bce_loss.cuda()
        cudnn.benchmark = True
    
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    
    # for alpha in FLAGS.alpha_list:
    eer_list = []
    epoch = 1
    num_of_batches = (len(train_label) // FLAGS.batch_size)
    loss_plot = []
    auc_list = []
    loss_per_epoch = 0
    txt_dir = 'output'
    save_dir = 'fc2_%s_%s_'%(FLAGS.split_type, FLAGS.save_dir)
    txt = '%s/ce_opl_%03d.txt'%(txt_dir, FLAGS.max_num_epoch)

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
            print('%s\tEpoch %03d'%(FLAGS.split_type, epoch))
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
            eer, auc = online_evaluation.test(FLAGS, model, i_test_data, j_test_data, test_label)


            
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


# i_train_batch,
# j_train_batch,
# batch_labels,
# model, optimizer, bce_loss)

def train(i_train_batch, j_train_batch, labels, model, optimizer, bce_loss):
    
    average_loss = RunningAverage()

    model.train()
    i_train_batch = torch.from_numpy(i_train_batch).float()
    j_train_batch = torch.from_numpy(j_train_batch).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        i_train_batch, j_train_batch, labels = i_train_batch.cuda(), i_train_batch.cuda(), labels.cuda()

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
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory for saving checkpoints.')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--max_num_epoch', type=int, default=500, help='Max number of epochs to train, number')
    parser.add_argument('--intermediate_emb', type=int, default=256, help='Intermediate Layer')
    parser.add_argument('--dim_embed', type=int, default=128, help='Embedding Size')
    parser.add_argument('--split_type', type=str, default='mfcc_only', help='split_type')

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    i_train_data, j_train_data, train_label = read_data('train', FLAGS)
    i_test_data, j_test_data, test_label = read_data('test', FLAGS)
    
    main(i_train_data, j_train_data, train_label, i_test_data, j_test_data, test_label)
