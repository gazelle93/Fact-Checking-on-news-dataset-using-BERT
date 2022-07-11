from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn

class BERTClassifier(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERTClassifier, self).__init__(config)
        self.bert = BertModel(config=config)
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim

        self.output_dim = 2
        self.batch_size = args.batch_size
        self.pad_len = args.pad_len

        self.tagger = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, _input):
        batch_input_ids = _input['input_ids'].reshape(self.batch_size, self.pad_len)
        batch_type_ids = _input['token_type_ids'].reshape(self.batch_size, self.pad_len)
        batch_attention_mask = _input['attention_mask'].reshape(self.batch_size, self.pad_len)

        sequence_output, pooled_output = self.bert(input_ids=batch_input_ids,
                                                   token_type_ids=batch_type_ids,
                                                   attention_mask=batch_attention_mask)

        output = self.tagger(pooled_output)
        return output

