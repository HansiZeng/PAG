import os 
import ujson 
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch 
import transformers 
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.file_utils import ModelOutput 

@dataclass
class T5ForLexicalSemanticOutput(ModelOutput):
    lexical_logit: torch.FloatTensor = None
    semantic_output: torch.FloatTensor = None 
    logits: torch.FloatTensor = None

@dataclass
class T5ForSemanticOutput(ModelOutput):
    semantic_output: torch.FloatTensor = None 
    logits: torch.FloatTensor = None 

class T5ForLexicalSemanticGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.mode = "lex_smt_outputs"
        assert self.mode in {"lex_smt_outputs", "lex_retrieval", "smt_retrieval"}, self.mode
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training_mode: Optional[str] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)


        #if self.config.tie_word_embeddings:
        #    raise NotImplementedError
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #    sequence_output = sequence_output * (self.model_dim**-0.5)

        ############### Hansi modification ########################
        #print("encoder_outputs: ", encoder_outputs.last_hidden_state.shape, sequence_output.shape)
        logits = None
        mode = self.mode if training_mode is None else training_mode
        if mode == "lex_smt_outputs":
            dec_tokens_length = sequence_output.size(1)
            enc_tokens_length = input_ids.size(1)
            assert dec_tokens_length - enc_tokens_length in {1, 8, 16}

            lexical_sequence = sequence_output[:, :enc_tokens_length].clone()
            if self.config.tie_word_embeddings:
                lexical_sequence = lexical_sequence * (self.model_dim**-0.5) 
            lexical_logits = self.lm_head(lexical_sequence)

            semantic_output = sequence_output[:, enc_tokens_length:].clone()
        elif mode == "lex_retrieval":
            assert sequence_output.size(1) == input_ids.size(1), (sequence_output.size(1), input_ids.size(1))

            lexical_sequence = sequence_output
            if self.config.tie_word_embeddings:
                lexical_sequence = lexical_sequence * (self.model_dim**-0.5) 
            lexical_logits = self.lm_head(lexical_sequence)
            semantic_output = None
        elif mode == "smt_retrieval":
            dec_tokens_length = sequence_output.size(1)
            enc_tokens_length = encoder_outputs.last_hidden_state.size(1)

            semantic_output = sequence_output[:, enc_tokens_length:].clone()
            lexical_logits = None
            logits = self.lm_head(semantic_output)
        elif mode == "smt_outputs":
            dec_tokens_length = sequence_output.size(1)
            enc_tokens_length = input_ids.size(1)
            assert dec_tokens_length - enc_tokens_length in {1} # this mode is only used for dense retrieval pretraining

            semantic_output = sequence_output[:, enc_tokens_length:].clone()
            lexical_logits = None
        else:
            raise NotImplementedError
        ###########################################################

        loss = None
        if labels is not None:
            raise NotImplementedError
            #loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            #labels = labels.to(lm_logits.device)
            #loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lexical_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return T5ForLexicalSemanticOutput(
            lexical_logit=lexical_logits, #[bz, enc_tokens_length, vocab_size]
            semantic_output=semantic_output, #[bz dec_token_length-enc_token_length, d_model]
            logits=logits #[bz dec_token_length-enc_token_length, d_model]
        )

class T5ForSemanticGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.return_logits = False
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        #if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #    sequence_output = sequence_output * (self.model_dim**-0.5)

        loss = None
        if labels is not None:
            raise NotImplementedError
            #loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            #labels = labels.to(lm_logits.device)
            #loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (sequence_output) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        if self.return_logits:
            return T5ForSemanticOutput(
                logits=self.lm_head(sequence_output),
                semantic_output=sequence_output
            )
        else:
            return T5ForSemanticOutput(
                semantic_output=sequence_output #[bz smtid_length, d_model]
            )
        
class Ripor(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__()

        config = T5Config.from_pretrained(model_name_or_path)
        if model_args is not None:
            config.num_decoder_layers = model_args.num_decoder_layers

        self.base_model = T5ForSemanticGeneration.from_pretrained(model_name_or_path, config=config)
        self.model_args = model_args 

    def forward():
        raise NotImplementedError 
    
    def decode(self, text_encodings):
        """
        Args:
            text_encodings: [bz, smtid_length]
        Returns:
            text_embeds: [bz, smtid_length, d_model]
        """
        text_embeds = torch.nn.functional.embedding(text_encodings, self.base_model.lm_head.weight)
        return text_embeds

    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

    def beam_search(self):
        raise NotImplementedError

class RiporForMarginMSE(Ripor):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        pos_query_embeds = self.base_model(**inputs["pos_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
        neg_query_embeds = self.base_model(**inputs["neg_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
        pos_doc_embeds = self.decode(inputs["pos_doc_encoding"]) #[bz, smtid_length, d_model]
        neg_doc_embeds = self.decode(inputs["neg_doc_encoding"]) #[bz, smtid_length, d_model]

        student_margin = (pos_query_embeds * pos_doc_embeds).sum(-1).sum(-1) - (neg_query_embeds * neg_doc_embeds).sum(-1).sum(-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        loss = self.loss_fn(student_margin, teacher_margin)

        return {"rank": loss}
    
class RiporForDirectLngKnpMarginMSE(Ripor):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        pos_query_embeds = self.base_model(**inputs["pos_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
        neg_query_embeds = self.base_model(**inputs["neg_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
        pos_doc_embeds = self.decode(inputs["pos_doc_encoding"]) #[bz, smtid_length, d_model]
        neg_doc_embeds = self.decode(inputs["neg_doc_encoding"]) #[bz, smtid_length, d_model]

        assert pos_doc_embeds.size(1) == 8, pos_doc_embeds.size()

        # rank_4 
        early_pos_score = (pos_query_embeds[:, :4, :].clone() * pos_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
        early_neg_score = (neg_query_embeds[:, :4, :].clone() * neg_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
        early_student_margin = early_pos_score - early_neg_score
        early_teacher_margin = (inputs["teacher_pos_scores"].clone() - inputs["teacher_neg_scores"].clone()) * 0.5
        rank_4_loss = self.loss_fn(early_student_margin, early_teacher_margin)

        # rank 
        student_margin = (pos_query_embeds * pos_doc_embeds).sum(-1).sum(-1) - (neg_query_embeds * neg_doc_embeds).sum(-1).sum(-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]
        rank_loss = self.loss_fn(student_margin, teacher_margin)

        return {
            "rank": rank_loss,
            "rank_4": rank_4_loss,
        }


    
    
class RiporForSeq2seq(Ripor):
    def __init__(self,  model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.rank_loss = torch.nn.CrossEntropyLoss()
        self.base_model.return_logits = True

    def forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, max_length] + [bz, smtid_length]
            labels: [bz, smtid_length]
        """
        logits = self.base_model(**inputs["tokenized_query"]).logits #[bz, smtid_length, vocab_size]

        bz, smtid_length = inputs["labels"].size()
        loss = self.rank_loss(logits.view(bz*smtid_length, -1), inputs["labels"].view(-1))

        return {
            "rank": loss 
        }

class RiporForLngKnpMarginMSE(Ripor):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)

        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, **inputs):
        """
        Args:
            pos_tokenized_query: [bz, seq_length] + [bz, smtid_length]
            neg_tokenized_query: [bz, seq_length] + [bz, smtid_length]
            pos_doc_encoding: [bz, smtid_length]
            neg_doc_encoding: [bz, smtid_length]

            teacher_pos_scores: [bz]
            teacher_neg_scores: [bz]

            smtid_8_teacher_pos_scores: [bz]
            smtid_8_teacher_neg_scores: [bz]
            smtid_16_teacher_pos_scores: [bz]
            smtid_16_teacher_neg_scores: [bz]
        """

        pos_query_embeds = self.base_model(**inputs["pos_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
        neg_query_embeds = self.base_model(**inputs["neg_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
        pos_doc_embeds = self.decode(inputs["pos_doc_encoding"]) #[bz, smtid_length, d_model]
        neg_doc_embeds = self.decode(inputs["neg_doc_encoding"]) #[bz, smtid_length, d_Model]
        
        # rank loss
        student_margin = (pos_query_embeds * pos_doc_embeds).sum(-1).sum(-1) - (neg_query_embeds * neg_doc_embeds).sum(-1).sum(-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]
        rank_loss = self.loss_fn(student_margin, teacher_margin)

        # rank_4 loss 
        early_pos_score = (pos_query_embeds[:, :4, :].clone() * pos_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
        early_neg_score = (neg_query_embeds[:, :4, :].clone() * neg_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
        early_margin = early_pos_score - early_neg_score
        early_teacher_margin = inputs["smtid_4_teacher_pos_scores"] - inputs["smtid_4_teacher_neg_scores"]
        rank_4_loss = self.loss_fn(early_margin, early_teacher_margin)

        if pos_doc_embeds.size()[1] == 8:
            return {"rank": rank_loss, "rank_4": rank_4_loss}
        elif pos_doc_embeds.size()[1] in [16, 32]:
            # rank_8 loss 
            early_pos_score = (pos_query_embeds[:, :8, :].clone() * pos_doc_embeds[:, :8, :].clone()).sum(-1).sum(-1)
            early_neg_score = (neg_query_embeds[:, :8, :].clone() * neg_doc_embeds[:, :8, :].clone()).sum(-1).sum(-1)
            early_margin = early_pos_score - early_neg_score
            early_teacher_margin = inputs["smtid_8_teacher_pos_scores"] - inputs["smtid_8_teacher_neg_scores"]
            rank_8_loss = self.loss_fn(early_margin, early_teacher_margin)

            if pos_doc_embeds.size()[1] == 16:
                return {"rank": rank_loss, "rank_8": rank_8_loss, "rank_4": rank_4_loss}
            elif pos_doc_embeds.size()[1] == 32:
                # rank_16 loss
                early_pos_score = (pos_query_embeds[:, :16, :].clone() * pos_doc_embeds[:, :16, :].clone()).sum(-1).sum(-1)
                early_neg_score = (neg_query_embeds[:, :16, :].clone() * neg_doc_embeds[:, :16, :].clone()).sum(-1).sum(-1)
                early_margin = early_pos_score - early_neg_score
                early_teacher_margin = inputs["smtid_16_teacher_pos_scores"] - inputs["smtid_16_teacher_neg_scores"]
                rank_16_loss = self.loss_fn(early_margin, early_teacher_margin)
                
                return {"rank": rank_loss, "rank_4": rank_4_loss, "rank_8": rank_8_loss, "rank_16": rank_16_loss}
            else:
                raise ValueError("not valid length: {}".format(pos_doc_embeds.size()[1]))
        else:
            raise ValueError("not valid length: {}".format(pos_doc_embeds.size()[1]))
        
class LexicalRipor(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__()

        config = T5Config.from_pretrained(model_name_or_path)
        if model_args is not None:
            config.num_decoder_layers = model_args.num_decoder_layers

        self.base_model = T5ForLexicalSemanticGeneration.from_pretrained(model_name_or_path, config=config)
        self.model_args = model_args 

    def forward():
        raise NotImplementedError 
    
    def rerank_forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, enc_token_length] + [bz, dec_token_length]
            doc_encoding: [bz, dec_token_length]
            lexical_encoding_size: int
        """
        assert self.base_model.mode == "lex_smt_outputs"

        query_lex_rep, query_smt_rep = self.encode(**inputs["tokenized_query"])

        lex_encoding_size = inputs["lexical_encoding_size"]
        lex_score = torch.sum(torch.gather(query_lex_rep, -1, 
                                            inputs["doc_encoding"][:,:lex_encoding_size].clone()), dim=-1)

        doc_smt_rep =  self.semantic_decode(inputs["doc_encoding"][:,lex_encoding_size:].clone())
        smt_score = (query_smt_rep * doc_smt_rep).sum(-1).sum(-1)

        return {
            "score": lex_score + smt_score
        }

    
    def semantic_decode(self, text_encodings):
        """
        Args:
            text_encodings: [bz, smtid_length]
        Returns:
            text_embeds: [bz, smtid_length, d_model]
        """
        text_embeds = torch.nn.functional.embedding(text_encodings, self.base_model.lm_head.weight)
        return text_embeds
    
    def encode(self, **inputs):
        """
        Returns:
            lexical_rep: [bz, vocab_size]
            semantic_rep: [bz, smtid_length, d_model]
        """
        model_output = self.base_model(**inputs, return_dict=True)
        lexical_rep, _ = torch.max(torch.log(1 + torch.relu(model_output.lexical_logit)) * inputs["attention_mask"].unsqueeze(-1), dim=1)
        semantic_rep = model_output.semantic_output 

        return lexical_rep, semantic_rep

    def encode_only_smt(self, **inputs):
        model_output = self.base_model(**inputs, return_dict=True)
        return model_output.semantic_output 

    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

    def beam_search(self):
        raise NotImplementedError
    
class LexicalRiporForMarginMSE(LexicalRipor):
    def __init__(self, model_name_or_path, model_args):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        pos_query_lexical, pos_query_semantic = self.encode(**inputs["pos_tokenized_query"])
        neg_query_lexical, neg_query_semantic = self.encode(**inputs["neg_tokenized_query"]) 

        lex_encoding_size = inputs["lexical_encoding_size"]
        pos_lexical_score = torch.sum(torch.gather(pos_query_lexical, -1, 
                                                   inputs["pos_doc_encoding"][:,:lex_encoding_size].clone()), dim=-1)
        neg_lexical_score = torch.sum(torch.gather(neg_query_lexical, -1, 
                                                   inputs["neg_doc_encoding"][:,:lex_encoding_size].clone()), dim=-1)

        pos_semantic_rep = self.semantic_decode(inputs["pos_doc_encoding"][:, lex_encoding_size:].clone())
        neg_semantic_rep = self.semantic_decode(inputs["neg_doc_encoding"][:, lex_encoding_size:].clone())

        pos_semantic_score = (pos_query_semantic * pos_semantic_rep).sum(-1).sum(-1)
        neg_semantic_score = (neg_query_semantic * neg_semantic_rep).sum(-1).sum(-1)

        lexical_margin = pos_lexical_score - neg_lexical_score 
        total_margin = (pos_lexical_score + pos_semantic_score) - (neg_lexical_score + neg_semantic_score)

        lexical_rank_loss = self.loss_fn(lexical_margin, (inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])*0.5)
        rank_loss = self.loss_fn(total_margin, inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])

        return {
            "lexical_rank": lexical_rank_loss,
            "rank": rank_loss 
        }
    
class LexicalRiporForDirectLngKnpMarginMSE(LexicalRipor):
    def __init__(self, model_name_or_path, model_args):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        pos_query_lexical, pos_query_semantic = self.encode(**inputs["pos_tokenized_query"])
        neg_query_lexical, neg_query_semantic = self.encode(**inputs["neg_tokenized_query"]) 

        # lexical
        lex_encoding_size = inputs["lexical_encoding_size"]
        pos_lexical_score = torch.sum(torch.gather(pos_query_lexical, -1, 
                                                   inputs["pos_doc_encoding"][:,:lex_encoding_size].clone()), dim=-1)
        neg_lexical_score = torch.sum(torch.gather(neg_query_lexical, -1, 
                                                   inputs["neg_doc_encoding"][:,:lex_encoding_size].clone()), dim=-1)
        lexical_margin = pos_lexical_score - neg_lexical_score 

        # semantic
        pos_semantic_rep = self.semantic_decode(inputs["pos_doc_encoding"][:, lex_encoding_size:].clone())
        neg_semantic_rep = self.semantic_decode(inputs["neg_doc_encoding"][:, lex_encoding_size:].clone())

        assert pos_semantic_rep.size(1) == 8, pos_semantic_rep.size()

        ## rank_4
        early_pos_semantic_score = (pos_query_semantic[:, :4, :].clone() * pos_semantic_rep[:, :4, :].clone()).sum(-1).sum(-1)
        early_neg_semantic_score = (neg_query_semantic[:, :4, :].clone() * neg_semantic_rep[:, :4, :].clone()).sum(-1).sum(-1)

        ## rank_8
        pos_semantic_score = (pos_query_semantic * pos_semantic_rep).sum(-1).sum(-1)
        neg_semantic_score = (neg_query_semantic * neg_semantic_rep).sum(-1).sum(-1)

        # old loss
        #total_early_margin = (pos_lexical_score + early_pos_semantic_score) - (neg_lexical_score + early_neg_semantic_score)
        #total_margin = (pos_lexical_score + pos_semantic_score) - (neg_lexical_score + neg_semantic_score)
        #lexical_rank_loss = self.loss_fn(lexical_margin, (inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])*0.5)
        #rank_4_loss = self.loss_fn(total_early_margin, (inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])*0.75)
        #rank_loss = self.loss_fn(total_margin, inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])
        
        # paper loss 
        lexical_rank_loss = self.loss_fn(lexical_margin, inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])
        rank_4_loss = self.loss_fn(early_pos_semantic_score - early_neg_semantic_score, 
                                   (inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])*0.5)
        rank_loss = self.loss_fn(pos_semantic_score - neg_semantic_score, 
                                 inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])

        return {
            "lexical_rank": lexical_rank_loss,
            "rank_4": rank_4_loss,
            "rank": rank_loss 
        }
    
class LexicalRiporForKLDiv(LexicalRipor):
    def __init__(self, model_name_or_path, model_args):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, **inputs):
        query_lexical, query_semantic = self.encode(**inputs["tokenized_query"])

        lex_encoding_size = inputs["lexical_encoding_size"]
        lexical_score = torch.sum(torch.gather(query_lexical, -1, 
                                                   inputs["doc_encoding"][:,:lex_encoding_size].clone()), dim=-1)

        semantic_rep = self.semantic_decode(inputs["doc_encoding"][:, lex_encoding_size:].clone())

        semantic_score = (query_semantic * semantic_rep).sum(-1).sum(-1)
        
        bz, nway = inputs["teacher_scores"].size()
        lexical_score = lexical_score.view(bz, nway)
        semantic_score = semantic_score.view(bz, nway)

        # lexical_loss
        teacher_probs_1 = torch.softmax(inputs["teacher_scores"]*0.5, dim=-1)
        lexical_rank_loss = self.loss_fn(lexical_score, teacher_probs_1)

        # semantic loss 
        teacher_probs_2 = torch.softmax(inputs["teacher_scores"], dim=-1)
        rank_loss =  self.loss_fn(lexical_score + semantic_score, teacher_probs_2)

        return {
            "lexical_rank": lexical_rank_loss,
            "rank": rank_loss 
        }

            
class LexicalRiporForDensePretrained(LexicalRipor):
    def __init__(self, model_name_or_path, model_args):
        super().__init__(model_name_or_path, model_args)
    
    def rerank_forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, query_length] + [bz, query_length+1]
            tokenized_doc: [bz, doc_length] + [bz, doc_length+1]
            doc_encoding: [bz, dec_token_length]
            lexical_encoding_size: int
        """
        assert self.base_model.mode == "lex_smt_outputs"

        query_lex_rep, query_smt_rep = self.encode(**inputs["tokenized_query"])

        lex_score = torch.sum(torch.gather(query_lex_rep, -1, 
                                            inputs["doc_encoding"][:,:-1].clone()), dim=-1)

        doc_smt_rep = self.base_model(**inputs["tokenized_doc"], return_dict=True, training_mode="smt_outputs").semantic_output 
        assert doc_smt_rep.size(1) == query_smt_rep.size(1) == 1, (doc_smt_rep.size(), query_smt_rep.size())

        query_smt_rep = query_smt_rep.squeeze(1)
        doc_smt_rep = doc_smt_rep.squeeze(1)

        smt_score = (query_smt_rep * doc_smt_rep).sum(-1)

        return {
            "score": lex_score + smt_score
        }
    
    def doc_encode(self, **inputs):
        rep = self.base_model(**inputs, return_dict=True, training_mode="smt_outputs").semantic_output 
        assert rep.size(1) == 1, (rep.size())

        return rep.squeeze(1)
    
    def query_encode(self, **inputs):
        return self.doc_encode(**inputs)
        
class LexicalRiporForDensePretrainedMarginMSE(LexicalRiporForDensePretrained):
    def __init__(self, model_name_or_path, model_args):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        """
        tokenized_query: [bz, query_length] + [bz, query_length+1]
        tokenized_pos_doc: [bz, doc_length] + [bz, doc_length+1]
        tokenized_neg_doc: [bz, doc_length] + [bz, doc_length+1]
        """
        #pos_query_lexical, pos_query_semantic = self.encode(**inputs["pos_tokenized_query"])
        #neg_query_lexical, neg_query_semantic = self.encode(**inputs["neg_tokenized_query"]) 
        #_, pos_doc_semantic = self.encode()

        query_lexical, query_semantic = self.encode(**inputs["tokenized_query"])
        pos_doc_semantic = self.base_model(**inputs["tokenized_pos_doc"], return_dict=True, training_mode="smt_outputs").semantic_output 
        neg_doc_semantic = self.base_model(**inputs["tokenized_neg_doc"], return_dict=True, training_mode="smt_outputs").semantic_output 

        pos_lexical_score = torch.sum(torch.gather(query_lexical, -1, 
                                                   inputs["pos_doc_encoding"][:,:-1].clone()), dim=-1)
        neg_lexical_score = torch.sum(torch.gather(query_lexical, -1, 
                                                   inputs["neg_doc_encoding"][:,:-1].clone()), dim=-1)
        
        assert query_semantic.size(1) == pos_doc_semantic.size(1) == neg_doc_semantic.size(1) == 1 
        query_rep = query_semantic.squeeze(1)
        pos_doc_rep = pos_doc_semantic.squeeze(1)
        neg_doc_rep = neg_doc_semantic.squeeze(1)

        pos_semantic_score = (query_rep * pos_doc_rep).sum(dim=-1)
        neg_semantic_score = (query_rep * neg_doc_rep).sum(dim=-1)

        lexical_margin = pos_lexical_score - neg_lexical_score 
        total_margin = (pos_lexical_score + pos_semantic_score) - (neg_lexical_score + neg_semantic_score)

        lexical_rank_loss = self.loss_fn(lexical_margin, (inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])*0.5)
        rank_loss = self.loss_fn(total_margin, inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])

        # the model v2 has the dense reank 
        semantic_margin = pos_semantic_score - neg_semantic_score
        dense_rank_loss = self.loss_fn(semantic_margin, (inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"])*0.5)

        return {
            "lexical_rank": lexical_rank_loss,
            "rank": rank_loss,
            "dense_rank": dense_rank_loss,
        }
    
class LexicalRiporForSeq2seq(LexicalRiporForDensePretrained):
    def __init__(self, model_name_or_path, model_args, apply_lex_loss=False):
        super().__init__(model_name_or_path, model_args)

        self.rank_loss = torch.nn.CrossEntropyLoss()
        if apply_lex_loss:
            self.lex_rank_loss = torch.nn.BCEWithLogitsLoss()
        self.apply_lex_loss = apply_lex_loss
    
    def forward(self, **inputs):
        """
        tokenized_query: [bz, seq_length] + [bz, seq_length+smtid_length]
        lex_encodings: [bz, 64]
        lex_neg_encodings: [bz, 256]
        smt_labels: [bz, smtid_length]
        """

        # binary bce loss for lexical
        #lex_labels = torch.zeros_like(lex_rep, device=lex_rep.device)
        #lex_labels.scatter_(1, inputs["lex_encodings"], 1)
        #lex_rank_loss = self.lex_rank_loss(lex_rep, lex_labels)
        
        # we use the new code to compute binary bce loss for lexical docid 
        if self.apply_lex_loss:
            lex_rep, smt_rep = self.encode(**inputs["tokenized_query"])
            pos_logits = torch.gather(lex_rep, 1, inputs["lex_encodings"]).clone().view(-1)
            neg_logits = torch.gather(lex_rep, 1, inputs["lex_neg_encodings"]).clone().view(-1)

            lex_labels = torch.cat((torch.ones_like(pos_logits, device=pos_logits.device),
                                    torch.zeros_like(neg_logits, device=neg_logits.device)))
            lex_logits = torch.cat((pos_logits, neg_logits))
            lex_rank_loss = self.lex_rank_loss(lex_logits, lex_labels)
        else:
            smt_rep = self.encode_only_smt(**inputs["tokenized_query"])

        # seq2seq loss for semantic
        bz, smtid_length = inputs["smt_labels"].size()
        smt_logits = self.base_model.lm_head(smt_rep)
        rank_loss = self.rank_loss(smt_logits.view(bz*smtid_length, -1), inputs["smt_labels"].view(-1))

        if self.apply_lex_loss:
            return {
                "lexical_rank": lex_rank_loss,
                "rank": rank_loss
            }
        else:
            return {
                "rank": rank_loss
            }
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None, apply_lex_loss=False):
        return cls(model_name_or_path, model_args, apply_lex_loss)





        
