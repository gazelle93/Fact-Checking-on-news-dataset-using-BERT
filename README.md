# Overview
- After the emergence of Attention, the language models leveraging the attention layer show the best performance in various NLP tasks. Attention allows attending to utilize the most relevant parts of the input sequence by leveraging the attention score which is a weighted result of all of the encoded input vectors simultaneously. Therefore, attention layers are able to increase the learning speed through parallelization without the restrictions appearing in such sequential architectures. BERT (Bidirectional Encoder Representations from Transformers) is a deep learning architecture based on 12 layers of Transformer Encoders. This project aims to implement text classification architecture using pre-trained language model BERT. Here, 'bert-base-uncased' is pre-trained on BookCorpus and English Wikipedia.


# Brief description
- dataset_preproess.py: Pre-processing raw dataset into the form of the content (text & author) and label.
- main.py: Parse the arguments and train the model or test the saved model.
- model.py: BERT model with a linear tagger layer.
- trainer.py: Load dataset and pre-process it with data loader. Load model and initialize the optimizer and loss function. Train the model and evaluate the model with dev and test dataset.
- utils.py: Split train, dev and test dataset. Load tokenizer and convert the pre-processed dataset into the form of BERT input. Report classification results in the form of precision, recall and f1-score.

# Prerequisites
- argparse
- torch
- pandas
- numpy
- argparse
- sklearn
- tqdm
- datetime
- datasets
- transformers

# Parameters
- batch_size(int, defaults to 1): Batch size.
- pad_len(int, defaults to 256): Padding length.
- learning_rate(float, defaults to 1e-5): Learning rate.
- num_epochs(int, defaults to 5): Number of epochs for training.
- strict(defaults to True): Selection of label pre-processing method (True: Strict, False: Lenient).
- ignore(defaults to True): Ignore the rare cases (True: Ignore, False: Use all).
- ignore_num(int, defaults to 5): Ignore rare cases if the occurrence is less than this number.
- add_author(defaults to True): Adding author to the text (True: author + text, False: text).
- training(defaults to True): True for training, False for observing the experiment result on test dataset of the saved model.
- saved_model(defaults to "model_20220709_220851_w_s_at_256"): 
- language_model(str, defaults to "bert-base-uncased"): Pre-trained Language Model. (bert-base-uncased, digitalepidemiologylab/covid-twitter-bert)

# References
- Attention: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- BERT: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- Fact check news Dataset: https://huggingface.co/datasets/datacommons_factcheck
