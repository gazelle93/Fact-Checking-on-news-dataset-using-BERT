from datasets import load_dataset
from utils import model_eval_clasfficiation_report, get_datalader
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from model import BERTClassifier
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import numpy as np
from datetime import datetime
import torch
from dataset_preprocess import revise_dataset
import os


def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("---Running on GPU.")
    else:
        device = torch.device('cpu')
        print("---Running on CPU.")

    # load dataset and pre-process it with data loader
    dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")
    selected_dataset = revise_dataset(dataset, _strict=args.strict, _ignore=args.ignore, _ignore_under=args.ignore_num)
    train_dataset_loader, dev_dataset_loader, test_dataset_loader = get_datalader(selected_dataset, args)

    # training
    # load model
    config = BertConfig.from_pretrained(args.language_model)
    model = BERTClassifier.from_pretrained(args.language_model, config=config, args=args).to(device)

    # get weight to handle imbalanced dataset
    class_weights=class_weight.compute_class_weight('balanced', ['True', 'Fake'], np.array([x['label'] for x in selected_dataset]))
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # timestamp for naming the saved model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_dev_loss = 1_000_000

    print("---Initiating training process.")
    train_loss_list = []
    dev_loss_list = []

    for epoch in range(args.num_epochs):
        loss_total = 0
        model.train()

        for _input in tqdm(train_dataset_loader):
            pred = model(_input[0].to(device))

            loss = loss_function(pred, _input[1].to(device))

            loss.backward(retain_graph=True)
            optimizer.step()
            model.zero_grad()

            loss_total+=loss.detach().item()

        print("Training Loss: {}".format(loss_total/len(train_dataset_loader)))

        train_loss_list.append(loss_total/len(train_dataset_loader))


        # Evaluation on dev dataset
        print("---Initiating evaluation process on dev dataset.")
        model.eval()

        dev_loss = 0
        prediction_list = []
        gold_list = []
        for _input in tqdm(dev_dataset_loader):
            pred = model(_input[0].to(device))
            loss = loss_function(pred, _input[1].to(device))

            dev_loss+=loss.detach().item()
            prediction_list.append(np.argmax(pred.detach().cpu().numpy(), axis=1))
            gold_list.append(np.argmax(_input[1].detach().numpy(), axis=1))

        print("Dev Loss: {}".format(dev_loss/len(dev_dataset_loader)))
        print(classification_report(np.array(gold_list).flatten(), np.array(prediction_list).flatten(), target_names=['True', 'Fake']))
        dev_loss_list.append(dev_loss/len(dev_dataset_loader))

        # Save model if this is the best model based on the development loss (this criteria can be f1-score)
        if best_dev_loss > dev_loss/len(dev_dataset_loader):
            print("---Save the current best model.")
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            best_dev_loss = dev_loss/len(dev_dataset_loader)

            if args.strict == True:
                _s = "s"
            else:
                _s = "l"

            if args.add_author == True:
                _ir = "at"
            else:
                _it = "t"

            model_path = './saved_models/model_{}_w_{}_{}_{}'.format(timestamp, _s, _ir, args.pad_len)
            print("---Model saved at {}.".format(model_path))
            torch.save(model.state_dict(), model_path)

    print("---Done training process.")

    # Evaluation on test dataset
    print("---Initiating evaluation process on test dataset.")
    model_eval_clasfficiation_report(model, test_dataset_loader)
    print("---Done evaluation process on test dataset.")


def saved_model_result(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("---Running on GPU.")
    else:
        device = torch.device('cpu')
        print("---Running on CPU.")


    # load dataset and pre-process it with data loader
    dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")
    selected_dataset = revise_dataset(dataset, _strict=False, _ignore=True, _ignore_under=5)
    _, _, test_dataset_loader = get_datalader(selected_dataset, args)

    # load saved model
    SAVED_PATH = "./saved_models/"+args.saved_model
    print("---Loading saved model from {}.".format(SAVED_PATH))
    config = BertConfig.from_pretrained(args.language_model)
    model = BERTClassifier.from_pretrained(args.language_model, config=config, args=args).to(device)
    model.load_state_dict(torch.load(SAVED_PATH))

    # evaluation result of the test dataset
    print("---Initiating evaluation process on test dataset.")
    model_eval_clasfficiation_report(model, test_dataset_loader, device)
    print("---Done evaluation process on test dataset.")
