import transformers
import torch
from torch import nn


class EmBert(nn.Module):

    def __init__(self, model_class=transformers.BertModel, pretrained_weights='bert-base-uncased', out_features=512,
                 average_outputs=True):
        super(EmBert, self).__init__()
        with torch.no_grad():
            self.model = model_class.from_pretrained(pretrained_weights)
        for param in self.model.parameters():
            param.requires_grad = False
        if out_features is None:
            self.fc = None
        else:
            self.fc = nn.Linear(self.model.config.hidden_size, out_features)
        self.average_outputs = average_outputs

    def forward(self, x, mask=None, average_outputs=None):
        average_outputs = self.average_outputs if average_outputs is None else average_outputs
        with torch.no_grad():
            x = self.model(x, attention_mask=mask < 2)[0]  # (N, L, 768)
        # (N, L, 768) --> (N, 768) -->  (N, F*K)  --> reshape (N, F, K)
        # (N, L, 768) --> (N, 768) --> P(K,768),
        if self.fc is not None:
            x = self.fc(x)  # (N, L, F) --> (N, L, F*K) --> reshape (N, L, F, K)
        if mask is not None:
            mask_ = mask < 1
            mask_ = mask_.unsqueeze(2)
            x = x * mask_

        if average_outputs:

            return self.average(x, mask)
        else:

            return x
    # def parameters(self, *args, **kwargs):
    #     return self.fc.parameters(*args, **kwargs)

    @staticmethod
    def average(x, mask=None):
        if mask is not None:
            mask = mask < 1
            mask = mask.unsqueeze(2)
            x = x * mask
            x = x.sum(dim=1) / torch.max(mask.sum(dim=1), torch.ones_like(mask.sum(dim=1)))
        else:
            x = x.mean(dim=1)  # (N, F)

        return x
    def get_depth(self):
        return self.model.config.hidden_size


def get_bert_model(out_features=512, average_outputs=True):
    model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer,
                                                        'bert-base-uncased')
    model = EmBert(model_class, pretrained_weights, out_features=out_features, average_outputs=average_outputs)

    return model


class BertTokenizer(transformers.BertTokenizer):
    def get_empty_token(self):
        empty_token = torch.tensor(self.encode('', add_special_tokens=True))
        mask = torch.ones(len(empty_token), dtype=torch.long)
        return empty_token, mask


def get_bert_tokenizer():
    tokenizer_class, pretrained_weights = (BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    return tokenizer


