import torch
from torch import nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        """
        BERT-based classifier for multi-label classification.

        Args:
            bert_model_name (str): Name of the pretrained BERT model.
            num_classes (int): Number of output classes.
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask.

        Returns:
            Tensor: Logits for each class.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
