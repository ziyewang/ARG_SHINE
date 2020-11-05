#bash argshine_prediction.sh fasta_file output_path PATH_to_interpro

fasta_file=$1
output_path=$2
PATH_to_interpro=$3

#run ARG-CNN
python ARG_CNN/ARG_CNN_predict.py --fasta_file ${fasta_file} --output_path ${output_path}

#run ARG-KNN
blastp -db training_database/coala90_top_16classes_for_component_train \
-query ${fasta_file} \
-evalue 1e-3 -num_threads 20 -outfmt 6 -max_target_seqs 30 \
-out ${output_path}/query_blast_protein_matches_fmt6.txt

python ARG_KNN/ARG_KNN_predict.py --fasta_file ${fasta_file} --output_path ${output_path}

#run ARG-InterPro
#generate interproscan output
bash ${PATH_to_interpro}/interproscan.sh -i ${fasta_file} -f json --outfile scripts/query_sequences.fasta.json

python scripts/binary_parser.py scripts/interpro.cfg

python ARG_InterPro/ARG_InterPro_predict.py --fasta_file ${fasta_file} --output_path ${output_path}

#gen blast identity score
python gen_identity_score.py --fasta_file ${fasta_file} --output_path ${output_path}

#run ARG-SHINE
python ARG_SHINE_predict.py --fasta_file ${fasta_file} --output_path ${output_path}



