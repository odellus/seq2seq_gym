from json import encoder
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import torch.nn as nn

def pull_encoder(model):
    return model.encoder

class OutputPooler(nn.Module):
    def __init__(self, hidden_size):
        super(OutputPooler, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()    

    def forward(self, x):
        x = torch.squeeze(x, axis = 1)
        x = self.fc(x)
        return self.act(x)

class T5EncoderPooledOutput(nn.Module):
    def __init__(self, encoder, hidden_size):
        super(T5EncoderPooledOutput, self).__init__()
        self.encoder = encoder
        self.output_pooler = OutputPooler(hidden_size=hidden_size)    
    
    def forward(self, x):
        x = self.encoder(input_ids=x).last_hidden_state
        x = torch.squeeze(
            x[:, 0:1, :], # Get first token for pooled output like BERT
            axis = 1,
        )
        return self.output_pooler(x)


class DualT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, rel_encoder, sent_encoder, decoder, config):
        super(DualT5ForConditionalGeneration, self).__init__(config=config)
        self.rel_encoder = rel_encoder
        self.sent_encoder = sent_encoder
        self.decoder = decoder
        self.hidden_size = self.decoder.config.d_model
        self.rel_pooled = T5EncoderPooledOutput(rel_encoder, self.hidden_size)
        self.sent_pooled = T5EncoderPooledOutput(sent_encoder, self.hidden_size)
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, rel_input_ids, sent_input_ids):
        rel_enc = self.rel_pooled(rel_input_ids)
        sent_enc = self.sent_pooled(sent_input_ids)
        res = torch.concat((rel_enc, sent_enc), axis = 1)
        res = self.fc(res)
        res = torch.unsqueeze(res, axis = 1)
        return self.decoder(inputs_embeds = res)

if __name__ == '__main__':
    pass