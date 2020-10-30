#########
#CNN
#########
#https://blog.csdn.net/sunny_xsc1994/article/details/82969867
#library imports
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


from Bio import SeqIO

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

LabelSmoothLoss_para=0.1

att_dropout_val =0.5
d_a=100
r=10

#r=10,20,30

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

def train_model(model, train_dl, val_dl, epochs=200, lr=0.001, model_path='',weight_decay=0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())  #
    optimizer = torch.optim.AdamW(parameters, lr=lr,weight_decay=weight_decay)
    best_acc = 0.0
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y in train_dl:
            x = x.long()  # why ？词的顺序
            y = y.long()
            y_pred = model(x.cuda())
            optimizer.zero_grad()
            loss =loss_fn(y_pred, y.cuda())
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        # val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        with torch.no_grad():
            val_loss, val_acc = validation_metrics(model, val_dl)
        print("#epoch %d, train loss %.4f, val loss %.4f and val accuracy %.4f" % (
        i, sum_loss / total, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc, e = val_acc,0
            torch.save(model.state_dict(), model_path)
        else:
            e +=1
            if early is not None and e > early:
                break



@torch.no_grad()
def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x.cuda())
        loss = F.cross_entropy(y_hat, y.cuda())
        pred = torch.max(y_hat, 1)[1]
        pred=pred.cpu()
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        # sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss / total, correct / total  # , sum_rmse/total

class embedding_CNN_attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_size, dropout_value=0.5,
                 MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, kernel_size=3, pool_kernel_size=3, stride=1,
                 channel_size=256,att_dropout_val=0.5,d_a=2000,r=500,drop_con1=0.0,drop_con2=0.0,drop_con3=0.0):
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
        # self.padding_idx = 0  # ??
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.pool_kernel_size = pool_kernel_size
        # self.fc = nn.Linear(in_features=channel_size,
        #                     out_features=class_size)

        self.fc = nn.Linear(channel_size * r, class_size)
        #self.fc2 = nn.Linear(channel_size, class_size)
        # self.pool_ = torch.nn.AdaptiveMaxPool1d(self.MAX_SEQUENCE_LENGTH)
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
        # yrh, modify lengths calculation position.
        # lengths = (x != self.padding_idx).sum(dim=-1)
        x = self.embeddings(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        xb = self.drop_con1(F.relu(self.conv1(x)))
        #xb = F.avg_pool1d(xb, self.pool_kernel_size)
        #xb = self.drop_con2(F.relu(self.conv2(xb)))
        #xb = F.avg_pool1d(xb, self.pool_kernel_size)
        out = xb
        #out = self.drop_con3(F.relu(self.conv3(xb)))
        out = out.permute(0, 2, 1)
        x = torch.tanh(self.linear_first(out))
        x = self.att_dropout(x)
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ out
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
        # out = self.fc(avg_sentence_embeddings)
        out = self.fc(sentence_embeddings.view(x.size(0), -1))
        #out = self.fc2(F.relu(out))
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


f_for_query='/mnt/Data/wzy/results/ismej_332_novel_protein_sequences.fasta'


query_texts = create_separate_sequence(f_for_query)

f_for_component_train='/mnt/Data/wzy/ARG_deepclassifier/coala_all_datasets/coala_process/'+dataset+'_for_component_train.fasta'

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
query_dataset = TensorDataset(query_x_tensor,torch.zeros(len(query_x_tensor)))  # create your datset
query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)



model_path ='/mnt/Data/wzy/results/'+model_name+'-'+dataset+'_bs_'+str(batch_size)+'_channelSize'+str(channel_size)+'_em'+str(embedding_dim)+'_lr'+str(learning_rate)+'_ks'+str(kernel_size)+'_pls'+str(pool_kernel_size)+'_stride'+str(stride)+'_drop'+str(dropout_value)+'_att_drop'+str(att_dropout_val)+'_d_a_'+str(d_a)+'_r'+str(r)+'_drop_con1_'+str(drop_con1)+'_drop_con2_'+str(drop_con2)+'_drop_con3_'+str(drop_con3)+'_n_epochs_'+str(n_epochs)+'_weight_decay_'+str(weight_decay)+'_LabelSmooth_'+str(LabelSmoothLoss_para)+'_net.pth'

model = embedding_CNN_attention(MAX_NB_CHARS, embedding_dim, class_num,dropout_value=dropout_value,MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,kernel_size=kernel_size,pool_kernel_size=pool_kernel_size,stride=stride,channel_size=channel_size,att_dropout_val=att_dropout_val,d_a=d_a,r=r,drop_con1=drop_con1,drop_con2=drop_con2,drop_con3=drop_con3)
model = model.cuda()

print(model_path)

model.load_state_dict(torch.load(model_path))


print(model_path)

predict_proba=get_probas(model, query_dataloader)


outfile = '/mnt/Data/wzy/results/ismej_332_novel_protein_sequences_final_CNN_predicted_proba.npy'
#outfile='/home/wangzy/code/ARG_classification_coala90/save_files/temp_results/'+model_name+'-'+dataset+'-test_proba.npy'
np.save(outfile, predict_proba)

