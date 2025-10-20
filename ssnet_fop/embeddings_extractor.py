
from __future__ import division
from __future__ import print_function

import argparse
import os

import torch
import torch.utils.data
import numpy as np
from torch.autograd import Variable

import pandas as pd
from binary_classification_model import SingleBranchWithDownproject, SingleBranchWithPadding


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def read_retrieval_data(i, feature_name):
    print(f'Reading feature {i}: {feature_name}')
    feature_file = f'../data/retrieval/id_{feature_name}_mmsr.tsv'

    features = pd.read_csv(feature_file, sep='\t')

    ids = features['id'].values
    features = features.set_index('id')

    features = np.asarray(features)
    return ids, features



def load_best(filename):
    filename = os.path.join(filename, 'checkpoint.pth.tar')
    model = torch.load(filename)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for training. Default is cuda, if cuda is not available, uses cpu',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--save_dir', type=str, default='model', help='Directory for saving checkpoints. Default model')
    parser.add_argument('--feature_i', type=str, default='mfcc_bow', help='feature i (first modality). Default mfcc_bow')
    parser.add_argument('--feature_j', type=str, default='mfcc_bow', help='feature j (second modality). Default mfcc_bow')
    parser.add_argument('--max_num_epoch', type=int, default=500, help='Max number of epochs to train, number. Default 500')
    parser.add_argument('--merging_technique', type=str, default='downproject', help='whether to downproject or pad if there is a dimension mismatch. Default downproject')
    parser.add_argument('--intermediate_emb', type=int, default=256, help='Intermediate Layer. Default 256')
    parser.add_argument('--dim_embed', type=int, default=128, help='Embedding Size. Default 128')

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available() and FLAGS.device == 'cuda':
        DEVICE = 'cuda'
        torch.cuda.manual_seed(FLAGS.seed)
        print('Running on CUDA')
    else:
        DEVICE = 'cpu'
        print('Running on CPU')

    # test_feat = read_data(FLAGS)
    # test(test_feat)

    fi_name, fj_name = FLAGS.feature_i, FLAGS.feature_j
    save_dir = f'fc2_{fi_name}_{fj_name}_{FLAGS.merging_technique}_{FLAGS.save_dir}'
    save_best = 'best_%s'%(save_dir)
    model_state_dict = load_best(save_best)
    model_state_dict = model_state_dict['state_dict']
    ids_i, numpy_data_i = read_retrieval_data('i', fi_name)
    ids_j, numpy_data_j = read_retrieval_data('j', fj_name)

    torch_data_i = torch.from_numpy(numpy_data_i).float()
    torch_data_j = torch.from_numpy(numpy_data_j).float()
    torch_data_i, torch_data_j = torch_data_i.to(DEVICE), torch_data_j.to(DEVICE)

    torch_data_i = Variable(torch_data_i)
    torch_data_j = Variable(torch_data_j)


    if FLAGS.merging_technique == 'padding' or torch_data_i.shape[1] == torch_data_j.shape[1]:
        model = SingleBranchWithPadding(FLAGS, torch_data_i.shape[1], torch_data_j.shape[1], DEVICE)
        print('padding')
    elif FLAGS.merging_technique == 'downprojection':
        model = SingleBranchWithDownproject(FLAGS, torch_data_i.shape[1], torch_data_j.shape[1], DEVICE)
    else:
        print(f'Merging technique {FLAGS.merging_technique} not recognized!')
    model.load_state_dict(model_state_dict)

    model.eval()
    model.to(DEVICE)

    embedded_i, embedded_j = model.embed_branch(torch_data_i, torch_data_j)
    embedded_i_df, embedded_j_df = pd.DataFrame(embedded_i.detach().cpu().numpy()), pd.DataFrame(embedded_j.detach().cpu().numpy())
    embedded_i_df.columns = [f'{i}_best_{save_dir}' for i in range(len(embedded_i_df.columns))]
    embedded_j_df.columns = [f'{i}_best_{save_dir}' for i in range(len(embedded_j_df.columns))]

    embedded_i_df.insert(0, 'id', ids_i)
    embedded_j_df.insert(0, 'id', ids_j)

    txt_dir = 'output'
    id_run = '%03d_f1_%s_f2_%s_%s.txt'%(FLAGS.max_num_epoch, fi_name, fj_name, FLAGS.merging_technique)
    run_out = '%s/%s'%(txt_dir, id_run)

    embedded_i_df.to_csv(f'{run_out}/f1_{fi_name}.tsv', sep='\t', index=False)
    embedded_j_df.to_csv(f'{run_out}/f2_{fi_name}.tsv', sep='\t', index=False)




