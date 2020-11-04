import pandas as pd
import numpy as np
from Bio import SeqIO
import argparse

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fasta_file', type=str, help=("The query protein sequences."))
    parser.add_argument('--output_path', type=str, help=(
        "The path for saving the output files."))
    args = parser.parse_args()

    return args

def gen_blast_identity_score(query_namelist, output_path):
    align_file = output_path+'/query_blast_protein_matches_fmt6.txt'
    identity_file = output_path + '/query_identity_score.npy'

    align_table = pd.read_csv(align_file, header=None, sep='\t')

    align_table_sorted = align_table.sort_values(by=[0, 2], ascending=(False, False))


    align_identity_dict = {}
    for i in range(len(align_table_sorted)):
        read_id = list(align_table_sorted[0][i:i + 1])[0]
        if read_id not in align_identity_dict:
            align_identity_dict[read_id] = list(align_table_sorted[2][i:i + 1])[0]

    identity_list = []
    for i in range(len(query_namelist)):
        if query_namelist[i] in align_identity_dict:
            identity_list.append(float(align_identity_dict[query_namelist[i]] / 100.0))
        else:
            identity_list.append(0.0)

    iden_score = np.array(identity_list)
    np.save(identity_file, iden_score)



if __name__ == '__main__':
    args = arguments()
    query_fasta = args.fasta_file #sys.argv[1]
    output_path = args.output_path#sys.argv[2]

    query_namelist=[]
    for seq_record in SeqIO.parse(query_fasta, 'fasta'):
        query_namelist.append(seq_record.id)
    gen_blast_identity_score(query_namelist, output_path)

