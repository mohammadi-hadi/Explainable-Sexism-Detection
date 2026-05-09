import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertModel,
    XLMRobertaTokenizer, XLMRobertaModel,
    DistilBertTokenizer, DistilBertModel
)


class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.xlm_roberta_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

        # Concatenating the 3 models: bert, xlm_roberta, distilbert
        self.fc = nn.Linear(768 * 3, num_labels)

    #def forward(self, input_ids):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        
        bert_output = self.bert_model(input_ids).last_hidden_state
        xlm_roberta_output = self.xlm_roberta_model(input_ids).last_hidden_state
        distilbert_output = self.distilbert_model(input_ids).last_hidden_state

        concatenated = torch.cat((bert_output, xlm_roberta_output, distilbert_output), dim=2)
        out = self.fc(concatenated[:, 0, :])
        return out


class CustomModelWithLoss(nn.Module):
    def __init__(self, model):
        super(CustomModelWithLoss, self).__init__()
        self.model = model

    def forward(self, inputs, attention_mask, target):
        outputs = self.model(inputs, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, target)
        loss.backward()
        return outputs, loss

def compute_attributions(model, input_ids, attention_mask, label):
    # Convert input_ids to tensor if not already and ensure correct shape
    input_ids = input_ids.unsqueeze(0).long()
    
    # Ensure attention_mask is a tensor and has the correct shape
    attention_mask = attention_mask.unsqueeze(0) if attention_mask is not None else None
    
    # Define baseline for input_ids and attention_mask
    baseline_ids = torch.zeros_like(input_ids).long()
    baseline_mask = torch.zeros_like(attention_mask, dtype=torch.uint8) if attention_mask is not None else None
    
    label = torch.tensor([label]).long()

    # Initialize Integrated Gradients
    integrated_gradients = IntegratedGradients(model)

    # Create a custom model with a custom forward pass
    custom_model = CustomModelWithLoss(model)

    # Compute attributions
    attributions = integrated_gradients.attribute(
        inputs=input_ids, 
        baselines=baseline_ids,
        target=label,
        additional_forward_args=(attention_mask, label),
        internal_batch_size=1,
        model=custom_model,
    )
    return attributions.squeeze(0)