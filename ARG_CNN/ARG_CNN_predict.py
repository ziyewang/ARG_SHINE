#########
#CNN
#########
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical


from Bio import SeqIO
import sys
import argparse

#from ARG_classification.features.get_labels import ar_name_to_label
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

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True

# defining the essential constant values
MAX_SEQUENCE_LENGTH = 2975
MAX_NB_CHARS = 21
amino_acids = list("ARNDCEQGHILKMFPSTWYV")


batch_size = 64

class_num=16
embedding_dim=64
learning_rate=0.001
kernel_size=20
pool_kernel_size=0
stride=1
weight_decay=0.0
dropout_value=0.5
channel_size=2048

drop_con1=0.5
drop_con2=0.5
drop_con3=0.5

early = None
model_name ='CNN_att_lsmooth_torch1.6_AdamW_onelayer_newAtt'
dataset='coala90_top_16classes'

n_epochs = 300

#LabelSmoothLoss_para=0.1

att_dropout_val =0.5
d_a=100
r=10

"""
class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

loss_fn = LabelSmoothLoss(LabelSmoothLoss_para)
"""
def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fasta_file', type=str, help=("The query protein sequences."))
    parser.add_argument('--output_path', type=str, help=(
        "The path for saving the output files."))
    args = parser.parse_args()

    return args

class embedding_CNN_attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_size, dropout_value=0.5,
                 MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, kernel_size=20, pool_kernel_size=0, stride=1,
                 channel_size=2048,att_dropout_val=0.5,d_a=100,r=100,drop_con1=0.0,drop_con2=0.0,drop_con3=0.0):
        super().__init__()
        # self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_value)
        self.drop_con1 = nn.Dropout(drop_con1)
        self.drop_con2 = nn.Dropout(drop_con2)
        self.drop_con3 = nn.Dropout(drop_con3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, channel_size, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size, stride=stride)

        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.pool_kernel_size = pool_kernel_size

        self.fc = nn.Linear(channel_size * r, class_size)
        self.linear_first = nn.Linear(channel_size, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.att_dropout= nn.Dropout(att_dropout_val)
        self.r = r

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        xb = self.drop_con1(F.relu(self.conv1(x)))
        out = xb
        out = out.permute(0, 2, 1)
        x = torch.tanh(self.linear_first(out))
        x = self.att_dropout(x)
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ out
        out = self.fc(sentence_embeddings.view(x.size(0), -1))
        return out


def create_separate_sequence(fa_file, amino_acids=amino_acids):
    print('Processing text dataset')
    texts = []
    for index, record in enumerate(SeqIO.parse(fa_file, 'fasta')):
        # tri_tokens = trigrams(record.seq)
        temp_str = ""
        for item in (record.seq):
            if item in amino_acids:
                temp_str = temp_str + " " + item
        texts.append(temp_str)
    return texts


# Creating labels for the protein sequences
def create_labels_for_sequences(sequences, fa_file, ar_name_to_label):
    sequences_labels = np.empty([len(sequences), 1])
    for index, record in enumerate(SeqIO.parse(fa_file, 'fasta')):
        ar_id = record.description.split('|')[-1]
        # print (family_id)
        sequences_labels[index] = ar_name_to_label[ar_id]
    return sequences_labels

@torch.no_grad()
def get_probas(model, valid_dl):
    model.eval()
    scores = []
    F_softmax = torch.nn.Softmax(dim=1)
    for x, y in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x.cuda())
        scores.append(F_softmax(y_hat.cpu()).numpy())

    return np.concatenate(scores)

def argcnn_predict(query_fasta,output_path,f_for_component_train):

    query_texts = create_separate_sequence(query_fasta)

    train_texts = create_separate_sequence(f_for_component_train)

    # finally, vectorize the text samples into a 2D integer tensor
    train_tokenizer = Tokenizer(num_words=MAX_NB_CHARS)
    train_tokenizer.fit_on_texts(train_texts)

    char_index = train_tokenizer.word_index
    print('Found %s unique tokens.' % len(char_index))

    query_sequences = train_tokenizer.texts_to_sequences(query_texts)

    query_x = pad_sequences(query_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    query_x_tensor = torch.from_numpy(query_x)  # [0:64*48] 0:32*97

    print (query_x_tensor.size())
    query_dataset = TensorDataset(query_x_tensor, torch.zeros(len(query_x_tensor)))  # create your datset
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

    model_path = 'models/components/CNN_2048_20_100_10.pth'
    model = embedding_CNN_attention(MAX_NB_CHARS, embedding_dim, class_num, dropout_value=dropout_value,
                                    MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, kernel_size=kernel_size,
                                    pool_kernel_size=pool_kernel_size, stride=stride, channel_size=channel_size,
                                    att_dropout_val=att_dropout_val, d_a=d_a, r=r, drop_con1=drop_con1,drop_con2=drop_con2,drop_con3=drop_con3
                                   )
    model = model.cuda()

    model.load_state_dict(torch.load(model_path))

    print(model_path)

    predict_proba = get_probas(model, query_dataloader)

    outfile = output_path+'/ARG_CNN_predicted_proba.npy'
    np.save(outfile, predict_proba)

    prediction_list=[]
    for i in range(len(predict_proba)):
        prediction_list.append(np.argmax(predict_proba[i]))

    pred_label = [clsid_to_class_name[i] for i in prediction_list]
    with open(
            output_path+'/ARG_CNN_predict_result.tsv',
            'w') as file_handle:
        for i in range(len(pred_label)):
            file_handle.write(query_namelist[i] + '\t' + pred_label[i] + '\n')


if __name__ == '__main__':
    args = arguments()
    query_fasta = args.fasta_file #sys.argv[1]
    #f_for_component_train_path = sys.argv[2]#'training_database'
    f_for_component_train='training_database/coala90_top_16classes_for_component_train.fasta'
    output_path = args.output_path#sys.argv[2]

    query_namelist=[]
    for seq_record in SeqIO.parse(query_fasta, 'fasta'):
        query_namelist.append(seq_record.id)


    print ("Starting ARG-CNN prediction!")
    argcnn_predict(query_fasta,output_path,f_for_component_train)
    print ("ARG-CNN prediction Finished!")
