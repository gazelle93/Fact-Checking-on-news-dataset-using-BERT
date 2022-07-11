import argparse
from trainer import train, saved_model_result

def main(args):
    if args.training == True:
        train(args)
    else:
        saved_model_result(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, help="Batch size.")
    parser.add_argument("--pad_len", default=256, help="Padding length.")
    parser.add_argument("--learning_rate", default=1e-5, help="Learning rate.")
    parser.add_argument("--num_epochs", default=5, help="Number of epochs for training.")
    parser.add_argument("--strict", default=True, help="Selection of label pre-processing method (True: Strict, False: Lenient).")
    parser.add_argument("--ignore", default=True, help="Ignore the rare cases (True: Ignore, False: Use all).")
    parser.add_argument("--ignore_num", default=5, help="Ignore rare cases if the occurrence is less than this number.")
    parser.add_argument("--add_author", default=True, help="Adding author to the text (True: author + text, False: text).")
    parser.add_argument("--training", default=True, help="True for training, False for observing the experiment result on test dataset of the saved model.")
    parser.add_argument("--saved_model", default="model_20220709_220851_w_s_at_256", help="Name of saved model. (The pre-processing method, input representation and padding length should be match with the other arguments)")
    parser.add_argument("--language_model", default="bert-base-uncased", help="Selection of the language model (BERT).")
    args = parser.parse_args()

    main(args)
