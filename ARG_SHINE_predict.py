import numpy as np
import xgboost

from itertools import chain
from collections import Counter
from tqdm import tqdm
import pickle
from sklearn import metrics
import os
import pandas as pd

def get_candidates(n, class_num):
    return [(i, j) for i in range(n) for j in range(class_num)]

def get_samples_add_iden(candidates, scores, identity_score,targets=None):
    f, y = [], []
    for i, c in tqdm(candidates):
        temp= [s_[i, c] for s_ in scores]
        temp.append(identity_score[i])
        f.append(temp)
        if targets is not None:
            y.append(1.0 if c == int(targets[i]) else 0.0)
    return np.asarray(f), np.asarray(y)

def get_samples(candidates, scores, targets=None):
    f, y = [], []
    for i, c in tqdm(candidates):
        f.append([s_[i, c] for s_ in scores])
        if targets is not None:
            y.append(1.0 if c == targets[i] else 0.0)
    return np.asarray(f), np.asarray(y)

class_num=16

model_over50_path='models/ensemble/ltr_CNN_KNN_blast_fillnawith0_over0_add_iden-coala90_top_16classes_ntree_80_Max_depth_11_subsample_0.1.pth'
with open(model_over50_path,'rb') as f1:
    classifier_over50 = pickle.load(f1)

model_0to50_path='models/ensemble/ltr_CNN_KNN_blast_interpro_fillnawith0_over0_add_iden-coala90_top_16classes_ntree_50_Max_depth_5_subsample_0.1.pth'
with open(model_0to50_path,'rb') as f2:
    classifier_0to50 = pickle.load(f2)

model_None_iden_path='models/ensemble/ltr_CNN_interpro_fillnawith0-coala90_top_16classes_ntree_2_Max_depth_4_subsample_0.55.pth'
with open(model_None_iden_path,'rb') as f3:
    classifier_None_iden = pickle.load(f3)

qurey_interpro_lr = np.load('/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/component_prob/interpro_lr-coala90_top_16classes-predict_proba.npy')
query_knn = np.load('/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/component_prob/KNN-diamond_top5_id20_e-10_softmax-ismej_332_novel_genes-predict_proba.npy')
query_cnn = np.load('/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/component_prob/ismej_332_novel_protein_sequences_final_CNN_predicted_proba.npy')

query_iden_score=np.load('/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/ismeJ_novel_genes_accession_list.tsv.blast.identity_score.npy')


prediction_list=[]
score_matrix=[]
for i in range(len(query_iden_score)):
    if query_iden_score[i]>0.5:
        query_candidates = get_candidates(1, class_num)
        test_x, test_y = get_samples_add_iden(query_candidates, (query_cnn[i:i+1], query_knn[i:i+1]),
                                              query_iden_score[i:i+1])
        d_test = xgboost.DMatrix(test_x)
        scores_ = classifier_over50.predict(d_test)
        score_matrix.append(scores_)
        y_pred = np.argmax(scores_)
        prediction_list.append(y_pred)
    elif query_iden_score[i]>0.0:
        query_candidates = get_candidates(1, class_num)
        test_x, test_y = get_samples_add_iden(query_candidates, (qurey_interpro_lr[i:i+1], query_cnn[i:i+1], query_knn[i:i+1]),
                                              query_iden_score[i:i+1])
        d_test = xgboost.DMatrix(test_x)
        scores_ = classifier_0to50.predict(d_test)
        score_matrix.append(scores_)
        y_pred = np.argmax(scores_)
        prediction_list.append(y_pred)
    else:
        query_candidates = get_candidates(1, class_num)
        test_x, test_y = get_samples(query_candidates, (qurey_interpro_lr[i:i+1], query_cnn[i:i+1]))
        d_test = xgboost.DMatrix(test_x)
        scores_ = classifier_None_iden.predict(d_test)
        score_matrix.append(scores_)
        y_pred = np.argmax(scores_)
        prediction_list.append(y_pred)


#####


ar_name_to_label = {'MULTIDRUG': 0,
                    'AMINOGLYCOSIDE': 1,
                    'MACROLIDE': 2,
                    'BETA-LACTAM': 3,
                    'GLYCOPEPTIDE': 4,
                    'TRIMETHOPRIM': 5,
                    'FOLATE-SYNTHESIS-INHABITOR': 6,
                    'TETRACYCLINE': 7,
                    'SULFONAMIDE': 8,
                    'FOSFOMYCIN': 9,
                    'PHENICOL': 10,
                    'QUINOLONE': 11,
                    'STREPTOGRAMIN': 12,
                    'BACITRACIN': 13,
                    'RIFAMYCIN': 14,
                    'MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN': 15}

clsid_to_class_name={ 0 :'MULTIDRUG',
                   1:'AMINOGLYCOSIDE',
                   2:'MACROLIDE',
                   3 :'BETA-LACTAM',
                   4 :'GLYCOPEPTIDE',
                   5 :'TRIMETHOPRIM',
                   6 :'FOLATE-SYNTHESIS-INHABITOR',
                   7 :'TETRACYCLINE',
                   8 : 'SULFONAMIDE',
                   9 :'FOSFOMYCIN',
                   10:'PHENICOL',
                   11:'QUINOLONE',
                   12:'STREPTOGRAMIN',
                   13:'BACITRACIN',
                   14:'RIFAMYCIN',
                   15:'MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN'}

pred_label = [clsid_to_class_name[i] for i in prediction_list]


namelist_file ='/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/ismeJ_novel_genes_accession_list.tsv'
namelist = pd.read_csv(namelist_file, header=None, sep='\t')

with open('/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/ARG_SHINE_predict_result.tsv','w') as file_handle:
    for i in range(len(pred_label)):
        file_handle.write(namelist[0][i]+'\t'+pred_label[i]+'\n')




outfile='//home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/component_prob/ARG_SHINE_query_score_matrix.npy'
np.save(outfile, score_matrix)
