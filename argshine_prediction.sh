#eg: ./run.sh <data_path>/input/strmgCAMI2_short_read_pooled_megahit_assembly.fasta 1000 4
# bash argshine_prediction.sh fasta_file output_path

fasta_file=$1
output_path=$2

#run ARG-CNN
python ARG_CNN/ARG_CNN_predict.py --fasta_file ${fasta_file} --output_path ${output_path}

#run ARG-KNN
blastp -db training_database/coala90_top_16classes_for_component_train \
-query ${fasta_file} \
-evalue 1e-3 -num_threads 20 -outfmt 6 -max_target_seqs 30 \
-out ${output_path}/query_blast_protein_matches_fmt6.txt

python ARG_KNN/ARG_KNN_predict.py --fasta_file ${fasta_file} --output_path ${output_path}

#run ARG-InterPro

#./interproscan.sh -i /home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/ismej_332_novel_protein_sequences.fasta -f json --outfile /home/wangzy/data/ARG_project/arg_project_result/coala90/validation_on_novel_genes/ismej_332_novel_genes/query_sequences.fasta.json

python scripts/binary_parser.py scripts/interpro.cfg

python ARG_InterPro/ARG_InterPro_predict.py --fasta_file ${fasta_file} --output_path ${output_path}

#gen blast identity score
python gen_identity_score.py --fasta_file ${fasta_file} --output_path ${output_path}
#ARG-SHINE
python ARG_SHINE_predict.py --fasta_file ${fasta_file} --output_path ${output_path}



