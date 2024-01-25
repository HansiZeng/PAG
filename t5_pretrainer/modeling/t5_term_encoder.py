import transformers
import torch 
import os 
from transformers import AutoModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Config, T5Model
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.activations import ACT2FN

from ..losses.regulariaztion import init_regularizer, L0

class T5DenseEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__()

        config = T5Config.from_pretrained(model_name_or_path)
        if model_args is not None:
            config.num_decoder_layers = model_args.num_decoder_layers
        self.base_model = T5Model.from_pretrained(model_name_or_path, config=config)
        self.model_args = model_args

        self.rank_loss = torch.nn.MSELoss()

    def forward(self, **inputs):
        raise NotImplementedError

    def encode(self, **inputs):
        hidden_state = self.base_model(**inputs, return_dict=True).last_hidden_state
        assert hidden_state.dim() == 3 and hidden_state.size(1) == 1

        return hidden_state.squeeze(1)

    def doc_encode(self, **inputs):
        return self.encode(**inputs)
    
    def query_encode(self, **inputs):
        return self.encode(**inputs)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

class T5DenseEncoderForMarginMSE(T5DenseEncoder):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)

        self.rank_loss = torch.nn.MSELoss()
    
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin, teacher_margin)

        return {
            "rank": rank_loss
        }


class T5TermEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args):
        super().__init__()
        config = T5Config.from_pretrained(model_name_or_path)
        if model_args is not None:
            config.num_decoder_layers = model_args.num_decoder_layers
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config)
        self.model_args = model_args
        #self.transform = T5PredictionHeadTransform(self.base_model.config)

    def forward(self):
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)


class T5Splade(T5TermEncoder):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.output_dim = self.base_model.config.vocab_size

    def encode(self, **inputs):
        assert "decoder_input_ids" in inputs, inputs.keys()
        seq_reps = self.base_model(**inputs, return_dict=True).logits #[bz, seq_length, dim]
        #sequence_output = encoder_outputs[0]
        #sequence_output = self.transform(sequence_output)
        #if self.base_model.config.tie_word_embeddings:
        #    sequence_output = sequence_output * (self.base_model.model_dim**-0.5)
        #seq_reps = self.base_model.lm_head(sequence_output) #[bz, seq_length, vocab_size]

        #print(seq_reps.shape)
        reps, _ = torch.max(torch.log(1 + torch.relu(seq_reps)) * inputs["attention_mask"].unsqueeze(-1), dim=1) #[bz, vocab_size]
        
        return reps
    
    def forward(self, **inputs):
        return self.encode(**inputs)

class T5SpaldeForMarginMSE(T5Splade):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)

        self.rank_loss = torch.nn.MSELoss()
        self.reg_loss = init_regularizer("FLOPS")

    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin, teacher_margin)
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = (self.reg_loss(pos_doc_rep) + self.reg_loss(neg_doc_rep)) / 2.

        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }
    
class T5TermEncoderForMarginMSE(T5Splade):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)

        self.rank_loss = torch.nn.MSELoss()
    
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]

        pos_doc_score = torch.sum(torch.gather(query_rep, -1, inputs["pos_doc_encoding"]), dim=-1)
        neg_doc_score = torch.sum(torch.gather(query_rep, -1, inputs["neg_doc_encoding"]), dim=-1)

        rank_loss = self.rank_loss(pos_doc_score - neg_doc_score, inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])

        return {
            "rank": rank_loss
        }
        
class BertTermEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__()

        self.base_model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.model_args = model_args

    def forward(self):
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

class BertSplade(BertTermEncoder):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.output_dim = self.base_model.config.vocab_size

    def encode(self, **inputs):
        seq_reps = self.base_model(**inputs)["logits"]
        #print("seq_reps: ", seq_reps.shape)

        reps, _ = torch.max(torch.log(1 + torch.relu(seq_reps)) * inputs["attention_mask"].unsqueeze(-1), dim=1) #[bz, vocab_size]
        
        return reps
    
    def forward(self, **inputs):
        raise NotImplementedError
    
class BertSpaldeForMarginMSE(BertSplade):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)

        self.rank_loss = torch.nn.MSELoss()
        self.reg_loss = init_regularizer("FLOPS")

    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin, teacher_margin)
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = (self.reg_loss(pos_doc_rep) + self.reg_loss(neg_doc_rep)) / 2.

        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }


class BertTermEncoderForMarginMSE(BertSplade):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)

        self.rank_loss = torch.nn.MSELoss()

    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]

        pos_doc_score = torch.sum(torch.gather(query_rep, -1, inputs["pos_doc_encoding"]), dim=-1)
        neg_doc_score = torch.sum(torch.gather(query_rep, -1, inputs["neg_doc_encoding"]), dim=-1)

        rank_loss = self.rank_loss(pos_doc_score - neg_doc_score, inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])

        return {
            "rank": rank_loss
        }
