from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class BertConditioningModel(nn.Module):
    def __init__(self, d_cond, bert_model_name="bert-base-uncased"):
        super(BertConditioningModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_cond_dim = self.bert.config.hidden_size  # 768 for BERT-base

        # Add an MLP to adapt BERT embeddings for conditioning
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.bert_cond_dim, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond)
        )

    def forward(self, text_descriptions, device):
        # Tokenize and process text through BERT
        tokens = self.tokenizer(text_descriptions, return_tensors="pt", padding=True, truncation=True).to(device)
        bert_embeddings = self.bert(**tokens).last_hidden_state[:, 0, :]  # Use CLS token
        cond = self.cond_mlp(bert_embeddings)
        return cond
