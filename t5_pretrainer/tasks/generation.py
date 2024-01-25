from typing import Any, Optional, Union, List, Callable, Iterable, Dict
import warnings
import inspect
import time


import torch
import torch.nn as nn 
import torch.distributed as dist
from transformers.generation_beam_search import BeamScorer
from transformers.generation_utils import (
    BeamSearchOutput, 
    BeamSearchEncoderDecoderOutput, 
    BeamSearchDecoderOnlyOutput, 
    GreedySearchOutput,
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    SampleOutput,
    BeamSampleOutput)
from transformers.pytorch_utils import torch_int_div
import scipy.sparse as sp
import numpy as np
import ujson

from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation_beam_search import BeamSearchScorer
from transformers.generation_beam_constraints import Constraint
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from tqdm import tqdm

logger = logging.get_logger(__name__)

def generate_for_constrained_prefix_beam_search(
                            model, 
                            inputs: Optional[torch.Tensor] = None,
                            max_length: Optional[int] = None,
                            min_length: Optional[int] = None,
                            do_sample: Optional[bool] = None,
                            early_stopping: Optional[bool] = None,
                            num_beams: Optional[int] = None,
                            temperature: Optional[float] = None,
                            top_k: Optional[int] = None,
                            top_p: Optional[float] = None,
                            typical_p: Optional[float] = None,
                            repetition_penalty: Optional[float] = None,
                            bad_words_ids: Optional[Iterable[int]] = None,
                            bos_token_id: Optional[int] = None,
                            pad_token_id: Optional[int] = None,
                            eos_token_id: Optional[int] = None,
                            length_penalty: Optional[float] = None,
                            no_repeat_ngram_size: Optional[int] = None,
                            encoder_no_repeat_ngram_size: Optional[int] = None,
                            num_return_sequences: Optional[int] = None,
                            max_time: Optional[float] = None,
                            max_new_tokens: Optional[int] = None,
                            decoder_start_token_id: Optional[int] = None,
                            use_cache: Optional[bool] = None,
                            num_beam_groups: Optional[int] = None,
                            diversity_penalty: Optional[float] = None,
                            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
                            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
                            constraints: Optional[List[Constraint]] = None,
                            output_attentions: Optional[bool] = None,
                            output_hidden_states: Optional[bool] = None,
                            output_scores: Optional[bool] = None,
                            return_dict_in_generate: Optional[bool] = None,
                            forced_bos_token_id: Optional[int] = None,
                            forced_eos_token_id: Optional[int] = None,
                            remove_invalid_values: Optional[bool] = None,
                            synced_gpus: Optional[bool] = False,
                            **model_kwargs,
                             ):
    assert prefix_allowed_tokens_fn is not None 
    model.eval()
    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    num_beam_groups = num_beam_groups if num_beam_groups is not None else model.config.num_beam_groups
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    if eos_token_id is None and hasattr(model.config, "decoder"):
        eos_token_id = model.config.decoder.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        # special case if pad_token_id is not defined
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids = model._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria
    # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
    if max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids.shape[-1]
    elif max_length is not None and max_new_tokens is not None:
        # Both are set, this is odd, raise a warning
        warnings.warn(
            "Both `max_length` and `max_new_tokens` have been set "
            f"but they serve the same purpose. `max_length` {max_length} "
            f"will take priority over `max_new_tokens` {max_new_tokens}.",
            UserWarning,
        )
    # default to config if still None
    max_length = max_length if max_length is not None else model.config.max_length

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # 6. determine generation mode
    is_constraint_gen_mode = constraints is not None
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_beam_sample_gen_mode = (
        (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    )
    is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    # 7. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        logits_processor=logits_processor,
    )

    # 8. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 9. beam search
    if num_return_sequences > num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

    if stopping_criteria.max_length is None:
        raise ValueError("`max_length` needs to be a stopping_criteria for now.")
    
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=model.device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )
    # 11. interleave input_ids with `num_beams` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids, expand_size=num_beams, is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs
    )

    # 12. run beam search
    return beam_search(
        model,
        input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        output_scores=output_scores,
        return_dict_in_generate=return_dict_in_generate,
        synced_gpus=synced_gpus,
        **model_kwargs,
    )

def beam_search(
        model,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        lexical_condition: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        #print("input_ids: ", input_ids)
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = None # pad_token_id if pad_token_id is not None else model.config.pad_token_id #######################
        eos_token_id = None # eos_token_id if eos_token_id is not None else model.config.eos_token_id #######################
        output_scores = output_scores if output_scores is not None else model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)


            #next_token_scores = nn.functional.log_softmax(
            #    next_token_logits, dim=-1
            #)  # (batch_size * num_beams, vocab_size)
            next_token_scores = next_token_logits

            if lexical_condition:
                next_token_scores_processed = logits_processor(input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):], 
                                                               next_token_scores)
            else:
                next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        #print("input_ids: ", input_ids.shape, beam_scores.shape)
        #print("encoder_outputs: ", model_inputs["encoder_outputs"].last_hidden_state.shape)
        #print("input_ids: ", input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):])
        #print(next_tokens.shape, next_indices.shape)
        #print(stopping_criteria.max_length)
        if lexical_condition:
            sequence_outputs = beam_scorer.finalize(
                #input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):],
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                #max_length=model_inputs["encoder_outputs"].last_hidden_state.size(1)+1
                max_length=stopping_criteria.max_length,
            ) 
        else:
            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=stopping_criteria.max_length,
            )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if model.config.is_encoder_decoder:
                if lexical_condition:
                    #print("sequences: ", sequence_outputs["sequences"][:, model_inputs["encoder_outputs"].last_hidden_state.size(1):])
                    return BeamSearchEncoderDecoderOutput(
                        sequences=sequence_outputs["sequences"][:, model_inputs["encoder_outputs"].last_hidden_state.size(1):],
                        sequences_scores=sequence_outputs["sequence_scores"],
                        scores=scores,
                        beam_indices=beam_indices,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                    )
                else:
                    return BeamSearchEncoderDecoderOutput(
                        sequences=sequence_outputs["sequences"],
                        sequences_scores=sequence_outputs["sequence_scores"],
                        scores=scores,
                        beam_indices=beam_indices,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                    )
            else:
                raise NotImplementedError
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


def generate_for_lexical_condition_beam_search(
                            model, 
                            inputs: Optional[torch.Tensor] = None,
                            decoder_input_ids: Optional[torch.Tensor] = None,
                            max_length: Optional[int] = None,
                            min_length: Optional[int] = None,
                            do_sample: Optional[bool] = None,
                            early_stopping: Optional[bool] = None,
                            num_beams: Optional[int] = None,
                            temperature: Optional[float] = None,
                            top_k: Optional[int] = None,
                            top_p: Optional[float] = None,
                            typical_p: Optional[float] = None,
                            repetition_penalty: Optional[float] = None,
                            bad_words_ids: Optional[Iterable[int]] = None,
                            bos_token_id: Optional[int] = None,
                            pad_token_id: Optional[int] = None,
                            eos_token_id: Optional[int] = None,
                            length_penalty: Optional[float] = None,
                            no_repeat_ngram_size: Optional[int] = None,
                            encoder_no_repeat_ngram_size: Optional[int] = None,
                            num_return_sequences: Optional[int] = None,
                            max_time: Optional[float] = None,
                            max_new_tokens: Optional[int] = None,
                            decoder_start_token_id: Optional[int] = None,
                            use_cache: Optional[bool] = None,
                            num_beam_groups: Optional[int] = None,
                            diversity_penalty: Optional[float] = None,
                            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
                            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
                            constraints: Optional[List[Constraint]] = None,
                            output_attentions: Optional[bool] = None,
                            output_hidden_states: Optional[bool] = None,
                            output_scores: Optional[bool] = None,
                            return_dict_in_generate: Optional[bool] = None,
                            forced_bos_token_id: Optional[int] = None,
                            forced_eos_token_id: Optional[int] = None,
                            remove_invalid_values: Optional[bool] = None,
                            synced_gpus: Optional[bool] = False,
                            **model_kwargs,
                             ):
    assert prefix_allowed_tokens_fn is not None 
    model.eval()
    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    num_beam_groups = num_beam_groups if num_beam_groups is not None else model.config.num_beam_groups
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    if eos_token_id is None and hasattr(model.config, "decoder"):
        eos_token_id = model.config.decoder.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        # special case if pad_token_id is not defined
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder: 
        #input_ids = model._prepare_decoder_input_ids_for_generation(
        #    batch_size,
        #    decoder_start_token_id=decoder_start_token_id,
        #    bos_token_id=bos_token_id,
        #    model_kwargs=model_kwargs,
        #)
        ######################## 
        input_ids = decoder_input_ids
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria
    # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
    if max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids.shape[-1]
    elif max_length is not None and max_new_tokens is not None:
        # Both are set, this is odd, raise a warning
        warnings.warn(
            "Both `max_length` and `max_new_tokens` have been set "
            f"but they serve the same purpose. `max_length` {max_length} "
            f"will take priority over `max_new_tokens` {max_new_tokens}.",
            UserWarning,
        )
    # default to config if still None
    max_length = max_length if max_length is not None else model.config.max_length

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # 6. determine generation mode
    is_constraint_gen_mode = constraints is not None
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_beam_sample_gen_mode = (
        (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    )
    is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    # 7. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        logits_processor=logits_processor,
    )

    # 8. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 9. beam search
    if num_return_sequences > num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

    if stopping_criteria.max_length is None:
        raise ValueError("`max_length` needs to be a stopping_criteria for now.")
    
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=model.device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )
    # 11. interleave input_ids with `num_beams` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids, expand_size=num_beams, is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs
    )

    # 12. run beam search
    return beam_search(
        model,
        input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        output_scores=output_scores,
        return_dict_in_generate=return_dict_in_generate,
        synced_gpus=synced_gpus,
        lexical_condition=True,
        **model_kwargs,
    )


def generate_for_lexical_inc_beam_search(
                            model, 
                            inputs: Optional[torch.Tensor] = None,
                            decoder_input_ids: Optional[torch.Tensor] = None,
                            lex_rescorer = None,
                            max_length: Optional[int] = None,
                            min_length: Optional[int] = None,
                            do_sample: Optional[bool] = None,
                            early_stopping: Optional[bool] = None,
                            num_beams: Optional[int] = None,
                            temperature: Optional[float] = None,
                            top_k: Optional[int] = None,
                            top_p: Optional[float] = None,
                            typical_p: Optional[float] = None,
                            repetition_penalty: Optional[float] = None,
                            bad_words_ids: Optional[Iterable[int]] = None,
                            bos_token_id: Optional[int] = None,
                            pad_token_id: Optional[int] = None,
                            eos_token_id: Optional[int] = None,
                            length_penalty: Optional[float] = None,
                            no_repeat_ngram_size: Optional[int] = None,
                            encoder_no_repeat_ngram_size: Optional[int] = None,
                            num_return_sequences: Optional[int] = None,
                            max_time: Optional[float] = None,
                            max_new_tokens: Optional[int] = None,
                            decoder_start_token_id: Optional[int] = None,
                            use_cache: Optional[bool] = None,
                            num_beam_groups: Optional[int] = None,
                            diversity_penalty: Optional[float] = None,
                            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
                            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
                            constraints: Optional[List[Constraint]] = None,
                            output_attentions: Optional[bool] = None,
                            output_hidden_states: Optional[bool] = None,
                            output_scores: Optional[bool] = None,
                            return_dict_in_generate: Optional[bool] = None,
                            forced_bos_token_id: Optional[int] = None,
                            forced_eos_token_id: Optional[int] = None,
                            remove_invalid_values: Optional[bool] = None,
                            synced_gpus: Optional[bool] = False,
                            **model_kwargs,
                             ):
    assert prefix_allowed_tokens_fn is not None 
    model.eval()
    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    num_beam_groups = num_beam_groups if num_beam_groups is not None else model.config.num_beam_groups
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    if eos_token_id is None and hasattr(model.config, "decoder"):
        eos_token_id = model.config.decoder.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        # special case if pad_token_id is not defined
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder: 
        #input_ids = model._prepare_decoder_input_ids_for_generation(
        #    batch_size,
        #    decoder_start_token_id=decoder_start_token_id,
        #    bos_token_id=bos_token_id,
        #    model_kwargs=model_kwargs,
        #)
        ######################## 
        input_ids = decoder_input_ids
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria
    # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
    if max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids.shape[-1]
    elif max_length is not None and max_new_tokens is not None:
        # Both are set, this is odd, raise a warning
        warnings.warn(
            "Both `max_length` and `max_new_tokens` have been set "
            f"but they serve the same purpose. `max_length` {max_length} "
            f"will take priority over `max_new_tokens` {max_new_tokens}.",
            UserWarning,
        )
    # default to config if still None
    max_length = max_length if max_length is not None else model.config.max_length

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # 6. determine generation mode
    is_constraint_gen_mode = constraints is not None
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_beam_sample_gen_mode = (
        (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    )
    is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    assert lex_rescorer._num_beams == num_beams, (lex_rescorer._num_beams, num_beams)
    # 7. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        logits_processor=logits_processor,
    )

    # 8. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 9. beam search
    if num_return_sequences > num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

    if stopping_criteria.max_length is None:
        raise ValueError("`max_length` needs to be a stopping_criteria for now.")
    
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=model.device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )
    # 11. interleave input_ids with `num_beams` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids, expand_size=num_beams, is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs
    )

    # 12. run beam search
    return beam_search_inc_lexical_score(
        model,
        input_ids,
        beam_scorer,
        lex_rescorer=lex_rescorer,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        output_scores=output_scores,
        return_dict_in_generate=return_dict_in_generate,
        synced_gpus=synced_gpus,
        lexical_condition=True,
        **model_kwargs,
    )

def beam_search_inc_lexical_score(
        model,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        lex_rescorer, 
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        lexical_condition: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        #print("input_ids: ", input_ids)
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = None # pad_token_id if pad_token_id is not None else model.config.pad_token_id #######################
        eos_token_id = None # eos_token_id if eos_token_id is not None else model.config.eos_token_id #######################
        output_scores = output_scores if output_scores is not None else model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)


            #next_token_scores = nn.functional.log_softmax(
            #    next_token_logits, dim=-1
            #)  # (batch_size * num_beams, vocab_size)
            next_token_scores = next_token_logits

            next_token_scores_processed = logits_processor(input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):], 
                                                            next_token_scores)
            
            # let's start to incorporate lexical_score 
            # input_ids: [bz*num_beams, cur_len]
            # next_token_scores_processed: [bz*num_beams, vocab_size]
            # beam_scores: [bz*num_beams]
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            lex_inc_scores = lex_rescorer(input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):], 
                                          next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            lex_inc_scores = lex_inc_scores.view(batch_size, num_beams * vocab_size)

            # let's use lex_inc_scores to find next_tokens
            # and use next_token_scores to get the scores 
            _, next_tokens = torch.topk(
                lex_inc_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_token_scores = torch.gather(next_token_scores, 1, next_tokens)

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        #print("input_ids: ", input_ids.shape, beam_scores.shape)
        #print("encoder_outputs: ", model_inputs["encoder_outputs"].last_hidden_state.shape)
        #print("input_ids: ", input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):])
        #print(next_tokens.shape, next_indices.shape)
        #print(stopping_criteria.max_length)
        
        sequence_outputs = beam_scorer.finalize(
            #input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):],
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            #max_length=model_inputs["encoder_outputs"].last_hidden_state.size(1)+1
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if model.config.is_encoder_decoder:
                #print("sequences: ", sequence_outputs["sequences"][:, model_inputs["encoder_outputs"].last_hidden_state.size(1):])
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"][:, model_inputs["encoder_outputs"].last_hidden_state.size(1):],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                raise NotImplementedError
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

def generate_for_lex_tmp_rescore(model, 
                            inputs: Optional[torch.Tensor] = None,
                            decoder_input_ids: Optional[torch.Tensor] = None,
                            logit_tmp_rescorer = None,
                            max_length: Optional[int] = None,
                            min_length: Optional[int] = None,
                            do_sample: Optional[bool] = None,
                            early_stopping: Optional[bool] = None,
                            num_beams: Optional[int] = None,
                            temperature: Optional[float] = None,
                            top_k: Optional[int] = None,
                            top_p: Optional[float] = None,
                            typical_p: Optional[float] = None,
                            repetition_penalty: Optional[float] = None,
                            bad_words_ids: Optional[Iterable[int]] = None,
                            bos_token_id: Optional[int] = None,
                            pad_token_id: Optional[int] = None,
                            eos_token_id: Optional[int] = None,
                            length_penalty: Optional[float] = None,
                            no_repeat_ngram_size: Optional[int] = None,
                            encoder_no_repeat_ngram_size: Optional[int] = None,
                            num_return_sequences: Optional[int] = None,
                            max_time: Optional[float] = None,
                            max_new_tokens: Optional[int] = None,
                            decoder_start_token_id: Optional[int] = None,
                            use_cache: Optional[bool] = None,
                            num_beam_groups: Optional[int] = None,
                            diversity_penalty: Optional[float] = None,
                            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
                            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
                            constraints: Optional[List[Constraint]] = None,
                            output_attentions: Optional[bool] = None,
                            output_hidden_states: Optional[bool] = None,
                            output_scores: Optional[bool] = None,
                            return_dict_in_generate: Optional[bool] = None,
                            forced_bos_token_id: Optional[int] = None,
                            forced_eos_token_id: Optional[int] = None,
                            remove_invalid_values: Optional[bool] = None,
                            synced_gpus: Optional[bool] = False,
                            **model_kwargs,
                             ):
    assert prefix_allowed_tokens_fn is not None 
    model.eval()
    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    num_beam_groups = num_beam_groups if num_beam_groups is not None else model.config.num_beam_groups
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    if eos_token_id is None and hasattr(model.config, "decoder"):
        eos_token_id = model.config.decoder.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        # special case if pad_token_id is not defined
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder: 
        #input_ids = model._prepare_decoder_input_ids_for_generation(
        #    batch_size,
        #    decoder_start_token_id=decoder_start_token_id,
        #    bos_token_id=bos_token_id,
        #    model_kwargs=model_kwargs,
        #)
        ######################## 
        input_ids = decoder_input_ids
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria
    # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
    if max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids.shape[-1]
    elif max_length is not None and max_new_tokens is not None:
        # Both are set, this is odd, raise a warning
        warnings.warn(
            "Both `max_length` and `max_new_tokens` have been set "
            f"but they serve the same purpose. `max_length` {max_length} "
            f"will take priority over `max_new_tokens` {max_new_tokens}.",
            UserWarning,
        )
    # default to config if still None
    max_length = max_length if max_length is not None else model.config.max_length

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # 6. determine generation mode
    is_constraint_gen_mode = constraints is not None
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
    is_beam_sample_gen_mode = (
        (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
    )
    is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    assert logit_tmp_rescorer._num_beams == num_beams, (logit_tmp_rescorer._num_beams, num_beams)
    # 7. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        logits_processor=logits_processor,
    )

     # 8. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 9. beam search
    if num_return_sequences > num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

    if stopping_criteria.max_length is None:
        raise ValueError("`max_length` needs to be a stopping_criteria for now.")
    
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=model.device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )
    # 11. interleave input_ids with `num_beams` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids, expand_size=num_beams, is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs
    )

    # 12. run beam search
    return beam_search_lex_tmp_rescore(
        model,
        input_ids,
        beam_scorer,
        logit_tmp_rescorer=logit_tmp_rescorer,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        output_scores=output_scores,
        return_dict_in_generate=return_dict_in_generate,
        synced_gpus=synced_gpus,
        lexical_condition=True,
        **model_kwargs,
    )

def beam_search_lex_tmp_rescore(
        model,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logit_tmp_rescorer, 
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        lexical_condition: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        #print("input_ids: ", input_ids)
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = None # pad_token_id if pad_token_id is not None else model.config.pad_token_id #######################
        eos_token_id = None # eos_token_id if eos_token_id is not None else model.config.eos_token_id #######################
        output_scores = output_scores if output_scores is not None else model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)


            #next_token_scores = nn.functional.log_softmax(
            #    next_token_logits, dim=-1
            #)  # (batch_size * num_beams, vocab_size)
            next_token_scores = next_token_logits

            next_token_scores_processed = logits_processor(input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):], 
                                                            next_token_scores) #[bz*num_beams, vocab_size]
            
            # let's get the temporary next_token_scores for topk selection
            tmp_next_token_scores = logit_tmp_rescorer(input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):],
                                                       next_token_scores_processed)

            tmp_next_token_scores = tmp_next_token_scores + beam_scores[:, None].expand_as(tmp_next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            #vocab_size = next_token_scores.shape[-1]
            #next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            vocab_size = tmp_next_token_scores.size(-1)
            
            tmp_next_token_scores = tmp_next_token_scores.view(batch_size, num_beams*vocab_size)
            next_token_scores = next_token_scores.view(batch_size, num_beams*vocab_size)

            _, next_tokens = torch.topk(
                tmp_next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        #print("input_ids: ", input_ids.shape, beam_scores.shape)
        #print("encoder_outputs: ", model_inputs["encoder_outputs"].last_hidden_state.shape)
        #print("input_ids: ", input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):])
        #print(next_tokens.shape, next_indices.shape)
        #print(stopping_criteria.max_length)

        sequence_outputs = beam_scorer.finalize(
            #input_ids[:, model_inputs["encoder_outputs"].last_hidden_state.size(1):],
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            #max_length=model_inputs["encoder_outputs"].last_hidden_state.size(1)+1
            max_length=stopping_criteria.max_length,
        ) 

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if model.config.is_encoder_decoder:  
                    #print("sequences: ", sequence_outputs["sequences"][:, model_inputs["encoder_outputs"].last_hidden_state.size(1):])
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"][:, model_inputs["encoder_outputs"].last_hidden_state.size(1):],
                    sequences_scores=sequence_outputs["sequence_scores"]*input_ids.size(-1), # we don't need average scores
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                raise NotImplementedError
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

