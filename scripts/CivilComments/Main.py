from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pandas as pd
import os
import datetime
import pkg_resources
# import seaborn as sns
import time
import scipy.stats as stats
import gc
import re
import operator
import sys
from sklearn import metrics
from progiter import ProgIter
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
# from torch.cuda import amp
from apex import amp
# from tornado import autoreload
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
from tqdm import tqdm, tqdm_notebook
import os
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
import warnings
# warnings.filterwarnings(action='once')
import pickle
# from apex import amp
import shutil



device=torch.device('cuda')
MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 3
cwd = os.getcwd()
Data_dir = cwd + "/input/jigsaw-unintended-bias-in-toxicity-classification"
Input_dir = cwd + "/input/"
WORK_DIR = Input_dir #cwd + "/working/"
num_to_load = 1000000                        #Train size to match time limit
valid_size = 100000                          #Validation Size
TOXICITY_COLUMN = 'target'

package_dir_a = "/input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam



print("check")
print(os.getcwd())
print("check2")
BERT_MODEL_PATH = cwd + '/input/bert-pretrained-models/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/'
# with open(BERT_MODEL_PATH+'vocab.txt', "r") as reader:
#     print(reader.readline())

filenames = next(os.walk(BERT_MODEL_PATH), (None, None, []))[2]  # [] if no file
print(filenames)
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
BERT_MODEL_PATH + 'bert_config.json',
WORK_DIR + 'pytorch_model.bin')


shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')


# This is the Bert configuration file
from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig(BERT_MODEL_PATH + 'bert_config.json')

# Converting the lines to BERT format
# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in ProgIter(example):
        # print(text)
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)



# %%time
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
train_df = pd.read_csv(os.path.join(Data_dir,"train.csv")).sample(num_to_load+valid_size,random_state=SEED)
print("Sample civil comments files : ", train_df)
print('loaded %d records' % len(train_df))

# Make sure all comment_text values are strings
train_df['comment_text'] = train_df['comment_text'].astype(str)

sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)
train_df=train_df.fillna(0)
# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns=['target']

train_df = train_df.drop(['comment_text'],axis=1)
# convert target to 0,1
train_df['target']=(train_df['target']>=0.5).astype(float)

X = sequences[:num_to_load]
y = train_df[y_columns].values[:num_to_load]
X_val = sequences[num_to_load:]
y_val = train_df[y_columns].values[num_to_load:]

test_df = train_df.tail(valid_size).copy()
train_df = train_df.head(num_to_load)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))

output_model_file = "scripts/CivilComments/results/bert_sample.bin"


lr = 2e-5
batch_size = 32
accumulation_steps=2
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model = BertForSequenceClassification.from_pretrained(WORK_DIR, cache_dir=None, num_labels=len(y_columns))
model.zero_grad()
model = model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
train = train_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

model = model.train()
# scaler = torch.cuda.amp.GradScaler(enabled=True)

# tq = tqdm.notebook.tqdm(range(EPOCHS))

for epoch in ProgIter(range(EPOCHS)):
    print("Epoch : ", epoch)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    # tk0 = tqdm.notebook.tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    optimizer.zero_grad()
    for i,(x_batch, y_batch) in enumerate(train_loader):
        print("step : ", i)
#        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss = F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
         # check with unscale optimizer
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        # opt.zero_grad()
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()# Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        # tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    # tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)

torch.save(model.state_dict(), output_model_file)



# Run validation
# The following 2 lines are not needed but show how to download the model for prediction
model = BertForSequenceClassification(bert_config, num_labels = len(y_columns))
model.load_state_dict(torch.load(output_model_file ))

model.to(device)
for param in model.parameters():
    param.requires_grad=False
model.eval()
valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

# tk0 = tqdm_notebook(valid_loader)
for i,(x_batch,)  in enumerate(valid_loader):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()



# From baseline kernel

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]>0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER =- 5, OVERALL_MODEL_WEIGHT = 0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)



SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>0.5]
    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


MODEL_NAME = 'model1'
test_df[MODEL_NAME] = torch.sigmoid(torch.tensor(valid_preds)).numpy()
TOXICITY_COLUMN = 'target'
bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, 'target')
print(bias_metrics_df)
auc_score = calculate_overall_auc(test_df, MODEL_NAME)
print("Overall auc : ", auc_score)
final_metric = get_final_metric(bias_metrics_df, auc_score)
print(final_metric)