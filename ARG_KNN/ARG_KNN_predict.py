###用deeparg的方法
#process_blast.py is from deeparg
##############
#make_alignments_json
from Bio import SeqIO
from tqdm import tqdm 

from scipy.special import softmax

import heapq
import numpy as np
import argparse

iden = 0
evalue = 1e-3
top_k=5

unique_label = ['MULTIDRUG', 'AMINOGLYCOSIDE', 'MACROLIDE',
                'BETA-LACTAM',
                'GLYCOPEPTIDE',
                'TRIMETHOPRIM',
                'FOLATE-SYNTHESIS-INHABITOR',
                'TETRACYCLINE',
                'SULFONAMIDE',
                'FOSFOMYCIN',
                'PHENICOL',
                'QUINOLONE',
                'STREPTOGRAMIN',
                'BACITRACIN',
                'RIFAMYCIN',
                'MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN']

mapObj_label = {'MULTIDRUG': 0,
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

def gen_top_k_alignment_matrix(alignment_matrix,top_k=5,score_flag=True):
    c = []
    for i in range(len(alignment_matrix)):
        b = heapq.nlargest(top_k, range(len(alignment_matrix[i])), alignment_matrix[i].take)
        c.append(b)
    temp = np.zeros((len(alignment_matrix),len(alignment_matrix[0])))
    for i in range(len(temp)):
        temp[i][c[i]] = 1
    top_k_alignment_matrix =alignment_matrix*temp
    if not score_flag:
        for i in range(len(top_k_alignment_matrix)):
            top_k_alignment_matrix[i][c[i]] = 1
    return top_k_alignment_matrix


def gen_alignments(glen,fname):
    # Features: database gene namelist

    Features = list(glen.keys())
    alignments = {}
    BHit = {}
    SF = {i: True for i in Features}
    # print SF
    #fname = ''
    print("traversing input file ...")
    BitScore = True
    if BitScore == True:
        measure = 11
    else:
        measure = 2

    for i in tqdm(open(fname)):
        i = i.strip().split("\t")
        if float(i[2]) < iden:
            continue  # if the alignment has an identity below the threshold
        if float(i[10]) > evalue:
            continue  # if the alignment has an evalue greater than the threshold

        try:
            if SF[i[1]]:
                alignments[i[0]].update({
                    i[1]: float(i[measure])
                })
        except:
            try:
                if SF[i[1]]:
                    alignments[i[0]] = {
                        i[1]: float(i[measure])
                    }
            except:
                pass

            # compute the best hit for each entry
        try:
            if SF[i[1]]:
                try:
                    if BHit[i[0]][1] < float(i[measure]):
                        BHit[i[0]] = [i[1], float(i[measure]), i]
                except Exception as e:
                    BHit[i[0]] = [i[1], float(i[measure]), i]
        except:
            pass

        # json.dump(alignments, open(fname+".BitScoreMatrix.json",'w'))
    return alignments


def argknn_predict(query_fasta,output_path,fname):
    glen_file = 'training_database/coala90_top_16classes_for_component_train_gene_len.tsv'
    glen = {i.split()[0]: float(i.split()[1]) for i in open(
        glen_file)}

    alignments = gen_alignments(glen,fname)
    print(len(alignments), " sequences passed the filters and ready for prediction")


    ###convert alignment to matrix

    database_gene_name_list = list(glen.keys())
    database_gene_name_label = [i.split('|')[-1] for i in database_gene_name_list]
    mapObj_name_to_label = dict(zip(database_gene_name_list, database_gene_name_label))

    mapObj = dict(zip(database_gene_name_list, range(len(database_gene_name_list))))

    # n*m matrix ;n: test size; m: database size
    alignment_matrix = np.zeros((len(alignments), len(database_gene_name_list)))

    gene_name_list = []
    i = 0
    for key in alignments:
        gene_name_list.append(key)
        for k in alignments[key]:
            alignment_matrix[i][mapObj[k]] = alignments[key][k]
        i += 1

    # m*c matrix; c: label size
    # unique_label = np.unique(database_gene_name_label)
    # mapObj_label = dict(zip(unique_label, range(len(unique_label))))


    database_to_label_matrix = np.zeros((len(database_gene_name_list), len(unique_label)))

    for i in range(len(database_to_label_matrix)):
        label = mapObj_label[mapObj_name_to_label[database_gene_name_list[i]]]
        database_to_label_matrix[i][label] = 1

    top_alignment_matrix = gen_top_k_alignment_matrix(alignment_matrix, top_k=top_k, score_flag=True)


    ##################
    # save_proba

    target_score_sum_per_label_matrix = top_alignment_matrix.dot(database_to_label_matrix)

    target_score_sum_per_label_matrix_normalize_softmax = softmax(target_score_sum_per_label_matrix, axis=1)

    class_num = 16


    seq_ID = []
    for seq_record in SeqIO.parse(query_fasta, 'fasta'):
        seq_ID.append(seq_record.id)

    mapObj_gene_to_number = dict(zip(gene_name_list, range(len(gene_name_list))))

    # class_weight=[263,844,563,4051,1638,424,1730,1448,217,102,318,154,11,90,15,48]
    # fill_na_value=np.array(class_weight)/np.array(class_weight).sum()
    fill_na_value = np.zeros(class_num)

    predict_proba = []
    for i in range(len(seq_ID)):
        if seq_ID[i] not in gene_name_list:
            predict_proba.append(fill_na_value)
        else:
            predict_proba.append(target_score_sum_per_label_matrix_normalize_softmax[mapObj_gene_to_number[seq_ID[i]]])

    predict_proba = np.array(predict_proba)

    outfile = output_path+'/ARG_KNN_predicted_proba.npy'
    np.save(outfile, predict_proba)

    ##save the result
    pred_label=[]
    for i in range(len(seq_ID)):
        if seq_ID[i] not in gene_name_list:
            pred_label.append('None')
        else:
            pred_label.append(clsid_to_class_name[np.argmax(predict_proba[i])])

    with open(
            output_path+'/ARG_KNN_predict_result.tsv',
            'w') as file_handle:
        for i in range(len(pred_label)):
            file_handle.write(query_namelist[i] + '\t' + pred_label[i] + '\n')



if __name__ == '__main__':
    args = arguments()
    query_fasta = args.fasta_file #sys.argv[1]
    output_path = args.output_path #sys.argv[2]

    query_namelist=[]
    for seq_record in SeqIO.parse(query_fasta, 'fasta'):
        query_namelist.append(seq_record.id)

    fname=output_path+'/query_blast_protein_matches_fmt6.txt'

    print ("Starting ARG-KNN prediction!")
    argknn_predict(query_fasta,output_path,fname)
    print ("ARG-KNN prediction Finished!")