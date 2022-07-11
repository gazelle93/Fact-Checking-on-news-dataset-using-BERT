from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

def split_train_dev_test(_dataset):
    # split dataset into train (80%), dev (10%), and test (10%)
    True_X = [{"author": x['author'], "text": x['text']} for x in _dataset if x['label'] == "True"]
    True_y = [{"label": x['label']} for x in _dataset if x['label'] == "True"]

    Fake_X = [{"author": x['author'], "text": x['text']} for x in _dataset if x['label'] == "Fake"]
    Fake_y = [{"label": x['label']} for x in _dataset if x['label'] == "Fake"]

    # Split Train & Dev + Test
    True_X_train, True_X_dev_test, True_y_train, True_y_dev_test = train_test_split(True_X, True_y, test_size=0.2, random_state=777)
    Fake_X_train, Fake_X_dev_test, Fake_y_train, Fake_y_dev_test = train_test_split(Fake_X, Fake_y, test_size=0.2, random_state=777)

    # Split Dev & Test
    True_X_dev, True_X_test, True_y_dev, True_y_test = train_test_split(True_X_dev_test, True_y_dev_test, test_size=0.5, random_state=777)
    Fake_X_dev, Fake_X_test, Fake_y_dev, Fake_y_test = train_test_split(Fake_X_dev_test, Fake_y_dev_test, test_size=0.5, random_state=777)
    
    # Shuffle Dataset
    Train_X, Train_y = shuffle(True_X_train + Fake_X_train, True_y_train + Fake_y_train, random_state=777)
    Dev_X, Dev_y = shuffle(True_X_dev + Fake_X_dev, True_y_dev + Fake_y_dev, random_state=777)
    Test_X, Test_y = shuffle(True_X_test + Fake_X_test, True_y_test + Fake_y_test, random_state=777)
    
    return Train_X, Train_y, Dev_X, Dev_y, Test_X, Test_y

def get_tokenizer(model_path):
    return BertTokenizer.from_pretrained(model_path)


def get_bert_input(tokenizer, X, y, args):
    _dataset_pair = []

    # input: [CLS] + AUTHOR + [SEP] + TEXT + [SEP]
    if args.add_author:
        for data_sample, label in zip(X, y):
            _author = data_sample['author']
            _text = data_sample['text']

            temp_X = tokenizer(_author + "[SEP]" + _text, return_tensors='pt', padding="max_length", max_length=args.pad_len)
            temp_Y = torch.zeros(2)

            if label['label'] == "True":
                temp_Y[0] = 1
            else:
                temp_Y[1] = 1

            _dataset_pair.append([temp_X, temp_Y])

        return _dataset_pair

    # input: [CLS] + TEXT + [SEP]
    for data_sample, label in zip(X, y):
        _text = data_sample['text']

        temp_X = tokenizer(_text, return_tensors='pt', padding="max_length", max_length=args.pad_len)

        temp_Y = torch.zeros(2)

        if label['label'] == "True":
            temp_Y[0] = 1
        else:
            temp_Y[1] = 1

        _dataset_pair.append([temp_X, temp_Y])

    return _dataset_pair


def get_datalader(_dataset, args):
    Train_X, Train_y, Dev_X, Dev_y, Test_X, Test_y = split_train_dev_test(_dataset)
    tokenizer = get_tokenizer("bert-base-uncased")

    train_dataset = get_bert_input(tokenizer, Train_X, Train_y, args)
    dev_dataset = get_bert_input(tokenizer, Dev_X, Dev_y, args)
    test_dataset = get_bert_input(tokenizer, Test_X, Test_y, args)

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=min(4, args.batch_size))
    dev_dataset_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=min(4, args.batch_size))
    test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=min(4, args.batch_size))

    return train_dataset_loader, dev_dataset_loader, test_dataset_loader


def model_eval_clasfficiation_report(_model, _dataset_loader, device):
    _model.eval()

    prediction_list = []
    gold_list = []
    for _input in tqdm(_dataset_loader):
        pred = _model(_input[0].to(device))

        prediction_list.append(np.argmax(pred.detach().cpu().numpy(), axis=1))
        gold_list.append(np.argmax(_input[1].detach().numpy(), axis=1))

    print(classification_report(np.array(gold_list).flatten(), np.array(prediction_list).flatten(), target_names=['True', 'Fake']))
