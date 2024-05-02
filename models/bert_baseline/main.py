import numpy as np
import pandas as pd
import torch
import sys
import argparse

from sklearn.metrics import classification_report as report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from bundle.scorers.scorer_subtask_3 import _read_csv_input_file, evaluate
from bundle.baselines.st3 import *

# unused imports:
# from venv import logger
# from tqdm import tqdm
# import os
# import sklearn
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn import svm
# from sklearn.multioutput import MultiOutputClassifier
# from transformers import XLMRobertaTokenizer, XLMRobertaModel
# from torch.optim.lr_scheduler import ExponentialLR


def main():
    
    parser = argparse.ArgumentParser(description='Subtask-3')
    parser.add_argument('train_folder',  type=str, nargs=1,
                        help='Path to training articles')
    parser.add_argument('dev_folder',  type=str, nargs=1,
                        help='Path to dev articles')
    parser.add_argument('train_labels',  type=str, nargs=1,
                    help='Path to training labels')
    parser.add_argument('dev_labels',  type=str, nargs=1,
                    help='Path to dev labels')
    parser.add_argument('classes_folder',  type=str, nargs=1,
                    help='Path to classes folder')
    parser.add_argument('dev_output',  type=str, nargs=1,
                    help='Path to dev_output file')
    parser.add_argument('output_true',  type=str, nargs=1,
                    help='Path to output_true file')
    parser.add_argument('dev_output_true',  type=str, nargs=1,
                    help='Path to dev_output_true file')
    parser.add_argument('-o', "--output",  type=str, nargs=1,
                help='Path to output predictions on dev (mandatory)')
    
    args = parser.parse_args()
    if not args.output:
        print("argument -o is mandatory")
        sys.exit(1)
    
    folder_train = args.train_folder[0]
    labels_train_fn = args.train_labels[0]

    folder_dev = args.dev_folder[0]
    labels_dev_fn = args.dev_labels[0]

    out_fn = args.output[0] 
    out_true = args.output_true[0]
    
    out_dev = args.dev_output[0]
    out_dev_true = args.dev_output_true[0]

    classes = args.classes_folder[0]

    labels = pd.read_csv(labels_train_fn,sep='\t',encoding='utf-8',header=None)
    labels = labels.rename(columns={0:'id',1:'line',2:'labels'})
    labels = labels.set_index(['id','line'])

    labels_dev = pd.read_csv(labels_dev_fn,sep='\t',encoding='utf-8',header=None)
    labels_dev = labels_dev.rename(columns={0:'id',1:'line',2:'labels'})
    labels_dev = labels_dev.set_index(['id','line'])
    
    train = make_dataframe(folder_train, labels_train_fn)
    dev = make_dataframe(folder_dev, labels_dev_fn)

    X_train = train['text'].values.tolist()
    Y_train = train['labels'].fillna('').str.split(',').values

    X_dev = dev['text'].values.tolist()
    Y_dev = dev['labels'].fillna('').str.split(',').values

    multibin= MultiLabelBinarizer()
    Y_train = multibin.fit_transform(Y_train)
    multibin_dev= MultiLabelBinarizer()
    Y_dev = multibin_dev.fit_transform(Y_dev)

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    loss_function = torch.nn.BCEWithLogitsLoss()  
    optimizer = Adam(model.parameters(), lr=2e-5)

    batch_size = 16
    total_loss = 0.0
    num_batches = 0
    num_epochs = 5
    num_classes = 20
    patience = 3
    best_loss = float('inf')

    for epoch in range(num_epochs):

        with open(out_fn, 'w') as file:
            pass
        with open(out_dev, 'w') as file:
            pass
        with open(out_true, 'w') as file:
            pass
        with open(out_dev_true, 'w') as file:
            pass

        total_loss = 0.0
        for i in range(0, 100, batch_size):
            model.train()

            batch_texts = X_train[i:i+batch_size]
            tokenized_input = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=300)
            
            with torch.no_grad():
                outputs = model(**tokenized_input)

            last_hidden_states = outputs.last_hidden_state
            
            Y_batch = Y_train[i:i+batch_size]
            Y_labels = train[i:i+batch_size]

            linear_layer = torch.nn.Linear(last_hidden_states.shape[-1], num_classes)
            classification_output = linear_layer(last_hidden_states) # print(classification_output.shape)

            sigmoid_output = torch.sigmoid(classification_output)
            copy_sig = sigmoid_output.clone()
            copy_sig[copy_sig >= 0.5] = 1
            copy_sig[copy_sig < 0.5] = 0
            #print("Y_batch_one_hot_labels:", Y_batch)
            #print(Y_batch.shape)
            #print(sigmoid_output.shape)

            sigmoid_o = copy_sig[:, 0, :]
            predicted_labels = sigmoid_o.int().tolist()
            #print("predicted labels: ", predicted_labels)

            out = multibin.inverse_transform(torch.tensor(predicted_labels))
            output = list(map(lambda x: ','.join(x), out))
            ou = pd.DataFrame(output, Y_labels.index)
            
            ou.to_csv(out_fn, sep='\t', header=None, mode='a')
            # ou.loc[first_indx:last_processed_index+1].to_csv(out_fn, sep='\t', header=None, mode='a')

            # print(Y_batch)
            # print(predicted_labels)
            f1_macro = f1_score(Y_batch, predicted_labels, average='macro', zero_division=0)
            f1_micro = f1_score(Y_batch, predicted_labels, average='micro', zero_division=0)
            print("F1 Macro:", f1_macro)
            print("F1 Micro:", f1_micro)

            loss = loss_function(sigmoid_o, torch.tensor(Y_batch, dtype=torch.float32))
            print("Batch loss:", loss, " for ", i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        ou = pd.read_csv(out_fn,sep='\t',encoding='utf-8',header=None)
        ou = ou.rename(columns={0:'id',1:'line',2:'labels'})
        ou = ou.set_index(['id','line'])

        # print(ou)
        copied_labels = labels.copy()
        for index in labels.index:
            mask = (ou.index == index)
            if mask.any():
                labels.at[index, 'labels'] = ou[mask].values[0]
            else:
                labels.at[index, 'labels'] = ''
            if index == ou.index[-1]:
                copied_labels = copied_labels.loc[:index].copy()
                copied_labels.to_csv(out_true, sep='\t', header=None)        
                break
        
        # labels.to_csv(out_fn, sep='\t', header=None, mode='a') # mode = 'a' ako zelimo nastaviti pisati odakle smo stali u .txt
        labels.loc[:index].copy().to_csv(out_fn, sep='\t', header=None)        

        average_loss = total_loss / (100 // batch_size)
        print("Epoch:", epoch + 1, "Average Batch Loss:", average_loss)

        '''
        pred_labels = _read_csv_input_file(out_fn)
        true_labels = _read_csv_input_file(out_true)
        macro_f1, micro_f1 = evaluate(pred_labels, true_labels, classes)
        print("Epoch:", epoch + 1, "F1 macro:", macro_f1, "F1 micro:", micro_f1)
        '''

        dev_loss = 0.0
        model.eval()
        for i in range(0, 100, batch_size):

            Y_labels = dev[i:i+batch_size]

            batch_texts = X_dev[i:i + batch_size]
            tokenized_input = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=300)
            
            outputs = model(**tokenized_input)
            last_hidden_states = outputs.last_hidden_state

            linear_layer = torch.nn.Linear(last_hidden_states.shape[-1], 19)
            classification_output = linear_layer(last_hidden_states)

            sigmoid_output2 = torch.sigmoid(classification_output)
            copy_sig2 = sigmoid_output2.clone()
            copy_sig2[copy_sig2 >= 0.5] = 1
            copy_sig2[copy_sig2 < 0.5] = 0

            sigmoid_o2 = copy_sig2[:, 0, :]

            predicted_labels_dev = sigmoid_o2.int().tolist()
            print("predicted labels: ", predicted_labels_dev)
            out_de = multibin_dev.inverse_transform(torch.tensor(predicted_labels_dev))
            #out_de = multibin_dev.inverse_transform(sigmoid_o2.detach().numpy())
            output_dev = list(map(lambda x: ','.join(x), out_de))
            ou_dev = pd.DataFrame(output_dev, Y_labels.index)
            ou_dev.to_csv(out_dev, sep='\t', header=None, mode='a')

            Y_devv = Y_dev[i:i + batch_size]
            Y_devv_float64 = np.array([np.array(y, dtype=np.float64) for y in Y_devv])
            loss = loss_function(sigmoid_o2, torch.tensor(Y_devv_float64, dtype=torch.float64))

            dev_loss += loss.item()

        ou_dev = pd.read_csv(out_dev,sep='\t',encoding='utf-8',header=None)
        ou_dev = ou_dev.rename(columns={0:'id',1:'line',2:'labels'})
        ou_dev = ou_dev.set_index(['id','line'])

        # print(ou_dev)
        copied_labels_dev = labels_dev.copy()
        for index_dev in labels_dev.index:
            mask_dev = (ou_dev.index == index_dev)
            if mask_dev.any():
                labels_dev.at[index_dev, 'labels'] = ou_dev[mask_dev].values[0]
            else:
                labels_dev.at[index_dev, 'labels'] = ''
            if index_dev == ou_dev.index[-1]:
                copied_labels_dev = copied_labels_dev.loc[:index_dev].copy()
                copied_labels_dev.to_csv(out_dev_true, sep='\t', header=None)        
                break
        
        # labels.to_csv(out_fn, sep='\t', header=None, mode='a') # mode = 'a' ako zelimo nastaviti pisati odakle smo stali u .txt
        labels_dev.loc[:index_dev].copy().to_csv(out_dev, sep='\t', header=None)        

        pred_labels = _read_csv_input_file(out_dev)
        true_labels = _read_csv_input_file(out_dev_true)
        macro_f1, micro_f1 = evaluate(pred_labels, true_labels, classes)
        print("Epoch:", epoch + 1, "F1 macro:", macro_f1, "F1 micro:", micro_f1)

        average_dev_loss = dev_loss / (len(X_dev) // batch_size)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss}, Dev Loss: {average_dev_loss}")

        if average_dev_loss < best_loss:
            best_loss = average_dev_loss
            early_stopping_counter = 0  
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Validation loss did not improve for", patience, "epochs. Early stopping...")
            break

    average_loss = total_loss / (len(X_train) // batch_size)
    print("Average Batch Loss:", average_loss)

    return average_loss
   

def test_model_on_dev_set(model, tokenizer, dev_data, multibin, batch_size, num_classes, out_fn):
    X_dev = dev_data['text'].values.tolist()
    Y_dev = dev_data['labels'].fillna('').str.split(',').values

    with open(out_fn, 'w') as file:
        pass

    for i in range(0, len(X_dev), batch_size):
        batch_texts = X_dev[i:i + batch_size]
        tokenized_input = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=300)

        with torch.no_grad():
            outputs = model(**tokenized_input)
        last_hidden_states = outputs.last_hidden_state

        linear_layer = torch.nn.Linear(last_hidden_states.shape[-1], num_classes)
        classification_output = linear_layer(last_hidden_states)

        sigmoid_output = torch.sigmoid(classification_output)
        copy_sig = sigmoid_output.clone()
        copy_sig[copy_sig >= 0.5] = 1
        copy_sig[copy_sig < 0.5] = 0

        sigmoid_o = copy_sig[:, 0, :]

        # Output predictions
        out = multibin.inverse_transform(sigmoid_o.detach().numpy())
        output = list(map(lambda x: ','.join(x), out))
        ou = pd.DataFrame(output)
        ou.to_csv(out_fn, sep='\t', header=None, mode='a')


if __name__ == "__main__":
    train_dtframe =  main()


def create_one_hot_vector(label):
    file_path = "../scorers/techniques_subtask3.txt"

    with open(file_path, "r") as file:
        persuasion_techniques = file.read().splitlines()
    one_hot_vector = [0] * len(persuasion_techniques)
    if label in persuasion_techniques:
        index = persuasion_techniques.index(label)
        one_hot_vector[index] = 1
    return one_hot_vector

def map_one_hot_to_labels(sigmoid_output):
    #file_path = "../scorers/techniques_subtask3.txt"
    file_path = "C:\\Users\\lucij\\Desktop\\dipl_2_sem\\apt\\projekt\\semeval2023task3bundle-v4\\scorers\\techniques_subtask3.txt"       
    with open(file_path, "r") as file:
        persuasion_techniques = file.read().splitlines()
    
    predicted_labels = []
    for prediction in sigmoid_output:
        labels = []
        for i, value in enumerate(prediction.detach().numpy()):
            if i == len(prediction.detach().numpy):
                break
            if (value == 1).any():
                labels.append(persuasion_techniques[i])
        predicted_labels.append(labels)
        print("Predicted labels:", labels)
    
    return predicted_labels