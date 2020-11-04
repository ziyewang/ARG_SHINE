from sklearn.linear_model.logistic import LogisticRegression
from collections import defaultdict
from Bio import SeqIO
from scipy.sparse import csr_matrix
import numpy as np

import pickle
import argparse

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

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fasta_file', type=str, help=("The query protein sequences."))
    parser.add_argument('--output_path', type=str, help=(
        "The path for saving the output files."))
    args = parser.parse_args()

    return args

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

def arginterpro_predict(query_fasta,output_path):
    #feature_file = '/home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/ismej_332_novel_genes_interpro_coala90_correct.txt'
    feature_file ='scripts/query_interpro_feature.txt'

    pid_feature = get_binary_feature(feature_file)

    query_seqs_ID, query_sequences_dict = gen_query_data(query_fasta)
    query_x = [pid_feature[seq] for seq in query_seqs_ID]

    row, col, data, n = [], [], [], 0
    for i in range(len(query_x)):
        for j in range(len(query_x[i])):
            row.append(n)
            col.append(int(query_x[i][j] - 1))
            data.append(float(1))
        n += 1

    query_x_binary = csr_matrix((data, (row, col)), shape=(n, num))

    file_handle = 'models/components/interpro_lr-coala90_top_16classes_Penalty_elasticnet_C_0.8_l1ratio_0.2.pth'

    classifier = pickle.load(open(file_handle, 'rb'))

    # save proba
    predict_proba = classifier.predict_proba(query_x_binary)

    outfile = output_path+'/ARG_interpro_predicted_proba.npy'
    np.save(outfile, predict_proba)

    prediction_list=[]
    for i in range(len(predict_proba)):
        prediction_list.append(np.argmax(predict_proba[i]))

    pred_label = [clsid_to_class_name[i] for i in prediction_list]
    with open(
            output_path+'/ARG_interpro_predict_result.tsv',
            'w') as file_handle:
        for i in range(len(pred_label)):
            file_handle.write(query_namelist[i] + '\t' + pred_label[i] + '\n')
            

if __name__ == '__main__':
    args = arguments()
    query_fasta = args.fasta_file #sys.argv[1]
    f_for_component_train='training_database/coala90_top_16classes_for_component_train.fasta'
    output_path = args.output_path#sys.argv[2]

    query_namelist=[]
    for seq_record in SeqIO.parse(query_fasta, 'fasta'):
        query_namelist.append(seq_record.id)


    print ("Starting ARG-InterPro prediction!")
    arginterpro_predict(query_fasta,output_path)
    print ("ARG-InterPro prediction Finished!")

