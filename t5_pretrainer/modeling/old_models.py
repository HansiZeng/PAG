import transformers
import torch 
import os 
from transformers import AutoModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Config, T5Model
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.activations import ACT2FN

from ..losses.regulariaztion import init_regularizer, L0

class T5PredictionHeadTransform(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.d_model, config.d_model)
        #self.transform_act_fn = ACT2FN[config.feed_forward_proj]
        #self.LayerNorm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        #hidden_states = self.transform_act_fn(hidden_states)
        #hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
