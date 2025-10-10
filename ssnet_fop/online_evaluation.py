
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

import pandas as pd
from sklearn import metrics
# from scipy.optimize import brentq
from sklearn.model_selection import KFold
from scipy import interpolate

def read_data():
    
    test_file_face = '../data/face/facenet_face_veriflist_test_random_unseenunheard.csv'
    test_file_voice = '../data/voice/voice_veriflist_test_random_unseenunheard.csv'

    print('Reading Test Faces')
    face_test = pd.read_csv(test_file_face, header=None)
    print('Reading Test Voices')
    voice_test = pd.read_csv(test_file_voice, header=None)
    
    face_test = np.asarray(face_test)[:,:512]
    voice_test = np.asarray(voice_test)[:,:512]
    
    test_list = []
    for dat in range(len(voice_test)):
        test_list.append(voice_test[dat])
        test_list.append(face_test[dat])
    
    test_list = np.asarray(test_list)
    test_feat = torch.from_numpy(test_list).float()
    
    # face_test = torch.from_numpy(face_test).float()
    # voice_test = torch.from_numpy(voice_test).float()
    return test_feat


# In[1]

def same_func(f):
    issame_lst = []
    for idx in range(len(f)):
        if idx % 2 == 0:
            issame = True
        else:
            issame = False
        issame_lst.append(issame)
    return issame_lst

def calculate_accuracy(threshold, sigmoids, labels):
    predict_issame = np.less(threshold, sigmoids)
    tp = np.sum(np.logical_and(predict_issame, labels))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(labels)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), labels))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / sigmoids.size
    return tpr, fpr, acc

def calculate_roc(thresholds, sigmoids, labels, nrof_folds=10):
    nrof_pairs = min(len(sigmoids), len(labels))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, sigmoids[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 sigmoids[test_set],
                                                                                                 labels[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], sigmoids[test_set],
                                                      labels[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def evaluate(sigmoids, labels, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, sigmoids,
                                       np.asarray(labels), nrof_folds=nrof_folds)
    
    print('\nEvaluating')
    return tpr, fpr, accuracy

def test(args, model, i_test_data, j_test_data, test_label):

    model.eval()
    model.cuda()

    i_test_data = torch.from_numpy(i_test_data).float()
    j_test_data = torch.from_numpy(j_test_data).float()
    if args.cuda:
        i_test_data = i_test_data.cuda()
        j_test_data = j_test_data.cuda()
    i_test_data = Variable(i_test_data)
    j_test_data = Variable(j_test_data)

    with torch.no_grad():
        sigmoids = model(i_test_data, j_test_data)
        
        # feat_list = feat_list.data
        # feat_list = feat_list.cpu().detach().numpy()
    
        print('Total Number of Samples: ', len(sigmoids))

        sigmoids = np.asarray(sigmoids.cpu())
    
        tpr, fpr, accuracy = evaluate(sigmoids, test_label, 10)
    
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        fnr = 1-tpr
        abs_diffs = np.abs(fpr-fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    #    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f\n\n' % eer)
    
    return eer, auc
