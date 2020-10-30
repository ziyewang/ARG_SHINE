#interpro lr wow
#'/home/wangzy/code/ARG_identification/result/interpro/interpro_coala70.txt' 这个文件包含所有序列的结果这样才不会乱，id才是对的
#之后别的数据如果要用的话得特殊处理一下？？可能是有点问题 8017-9902 虽然这部分特征没有训练，也不好扩展

#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from collections import defaultdict
from Bio import SeqIO
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

from sklearn import metrics
import pickle

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

def get_binary_feature(feature_file):
    feature = defaultdict(list)
    with open(feature_file) as fp:
        for line in fp:
            pid, *f_ = line.split()
            if f_:
                feature[pid] += [int(x.split(':')[0]) + 1 for x in f_]
    return feature

def gen_data(fasta_file,ar_name_to_label):
    sequences = {}
    seq_ID = []
    class_name = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        sequences[seq_record.id] = seq_record.seq
        temp = seq_record.id.split('|')
        class_name.append(ar_name_to_label[temp[-1]])
        seq_ID.append(seq_record.id)
    return seq_ID,class_name,sequences

def gen_query_data(fasta_file):
    sequences = {}
    seq_ID = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        sequences[seq_record.id] = seq_record.seq
        seq_ID.append(seq_record.id)
    return seq_ID,sequences

model_name = 'interpro_lr'
dataset = 'coala90_top_16classes'
class_num = 16

num= 8240

feature_file ='/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/ismej_332_novel_genes_interpro_coala90_correct.txt'
#fasta_file='/home/wangzy/code/ARG_identification/ARG_deepclassifier/coala_all_datasets/coala_process/coala70_for_component_train.fasta'

#f_for_component_train='/home/wangzy/code/ARG_identification/ARG_deepclassifier/coala_all_datasets/coala_process/'+dataset+'_for_component_train.fasta'
f_for_query='/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/ismej_332_novel_protein_sequences.fasta'


pid_feature = get_binary_feature(feature_file)

query_seqs_ID,query_sequences_dict = gen_query_data(f_for_query)
query_x = [pid_feature[seq] for seq in query_seqs_ID]

row, col, data, n= [], [], [], 0
for i in range(len(query_x)):
    for j in range(len(query_x[i])):
        row.append(n)
        col.append(int(query_x[i][j] - 1))
        data.append(float(1))
    n +=1

#wc -l table_coala70.txt
query_x_binary=csr_matrix((data, (row, col)), shape=(n, num))

#LogisticRegression同样实现了fit()和predict()方法


#classifier.fit(train_x_binary,train_y)
#file_handle='/home/wangzy/code/ARG_classification/save_files/models/'+model_name+'-'+dataset+'.pth'
file_handle='/home/wangzy/data/ARG_project/arg_project_result/coala90/ours/ARG_SHINE/models/components/interpro_lr-coala90_top_16classes_Penalty_elasticnet_C_0.8_l1ratio_0.2.pth'

classifier=pickle.load(open(file_handle, 'rb'))

predictions=classifier.predict(query_x_binary)

#存预测概率
query_proba=classifier.predict_proba(query_x_binary)
outfile='/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/component_prob/'+model_name+'-'+dataset+'-predict_proba.npy'
np.save(outfile, query_proba)


