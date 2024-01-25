import os 

import wandb
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from torch.utils.data.dataloader import DataLoader
import ujson
import torch
from copy import deepcopy

from .modeling.t5_term_encoder import (
    T5Splade,
    T5SpaldeForMarginMSE,
    BertSpaldeForMarginMSE,
    BertTermEncoderForMarginMSE,
    T5DenseEncoderForMarginMSE,
    T5TermEncoderForMarginMSE,
)

from .modeling.t5_generative_retriever import (
    RiporForMarginMSE, 
    LexicalRiporForMarginMSE,
    RiporForSeq2seq,
    RiporForLngKnpMarginMSE,
    LexicalRiporForDensePretrainedMarginMSE,
    LexicalRiporForSeq2seq,
    LexicalRiporForKLDiv,
    RiporForDirectLngKnpMarginMSE,
    LexicalRiporForDirectLngKnpMarginMSE,
)

from .dataset.dataset import (
    MarginMSEDataset,
    CollectionDatasetPreLoad,
    TermEncoderForMarginMSEDataset,
    RiporForMarginMSEDataset,
    LexicalRiporForMarginMSEDataset,
    RiporForSeq2seqDataset,
    RiporForLngKnpMarginMSEDataset,
    LexicalRiporForDensePretrainedMarginMSEDataset,
    LexicalRiporForSeq2seqDataset,
    LexicalRiporForKLDivDataset
)
from .dataset.data_collator import (
    MarginMSECollator, 
    TermEncoderForMarginMSECollator,
    T5SpladeMarginMSECollator,
    T5DenseMarginMSECollator,
    T5TermEncoderForMarginMSECollator,
    RiporForMarginMSECollator,
    LexicalRiporForMarginMSECollator,
    RiporForSeq2seqCollator,
    RiporForLngKnpMarginMSECollator,
    LexicalRiporForDensePretrainedMarginMSECollator,
    LexicalRiporForSeq2seqCollator,
    LexicalRiporForKLDivCollator
)
from .dataset.dataloader import CollectionDataLoader

from .tasks.splade_trainer import SpladeTrainer, Splade_TrainingArgs, TermEncoder_TrainingArgs

from .arguments import ModelArguments, Arguments
from .losses.regulariaztion import RegWeightScheduler
from .tasks.evaluator import SparseApproxEvalWrapper
from .utils.utils import is_first_worker

MODEL_TYPES = {"bert_splade", "t5_splade", "bert_term_encoder", "t5_term_encoder", "t5_dense", "ripor", "lexical_ripor",
               "ripor"}
LOSS_TYPE = {"margin_mse", "constrastive", "seq2seq", "lng_knp_margin_mse", "kl_div",
             "direct_lng_knp_margin_mse"}
model_cls_dict = {"t5_splade": {"margin_mse": T5SpaldeForMarginMSE},
                  "bert_splade": {"margin_mse": BertSpaldeForMarginMSE},
                  "bert_term_encoder": {"margin_mse": BertTermEncoderForMarginMSE},
                  "t5_term_encoder": {"margin_mse": T5TermEncoderForMarginMSE},
                  "t5_dense": {"margin_mse": T5DenseEncoderForMarginMSE},
                  "ripor": {"margin_mse": RiporForMarginMSE, "seq2seq": RiporForSeq2seq,
                            "lng_knp_margin_mse": RiporForLngKnpMarginMSE,
                            "direct_lng_knp_margin_mse": RiporForDirectLngKnpMarginMSE},
                  "lexical_ripor": {"margin_mse": LexicalRiporForMarginMSE,
                                    "kl_div": LexicalRiporForKLDiv,
                                    "dense_pretrained_margin_mse": LexicalRiporForDensePretrainedMarginMSE,
                                    "seq2seq": LexicalRiporForSeq2seq,
                                    "direct_lng_knp_margin_mse": LexicalRiporForDirectLngKnpMarginMSE}}


def initialize_splade_evaluator(model, args, model_args):
    eval_collection_dataset = CollectionDatasetPreLoad(args.eval_collection_path, id_style="content_id")
    eval_collection_loader = CollectionDataLoader(
                                        dataset=eval_collection_dataset, 
                                        tokenizer_type=model_args.model_name_or_path,
                                        max_length=args.max_length,
                                        batch_size=32,
                                        num_workers=1)
    
    eval_query_dataset = CollectionDatasetPreLoad(args.eval_queries_path, id_style="content_id")
    eval_query_loader = CollectionDataLoader(
                                        dataset=eval_query_dataset, 
                                        tokenizer_type=model_args.model_name_or_path,
                                        max_length=args.max_length,
                                        batch_size=1,
                                        num_workers=1)
    
    eval_config = {
        "top_k": args.full_rank_eval_topk,
        "index_dir": args.full_rank_index_dir,
        "out_dir": args.full_rank_out_dir
    }

    splade_evaluator = SparseApproxEvalWrapper(model, eval_config, eval_collection_loader, eval_query_loader)
    return splade_evaluator, eval_config

def main():
    parser = HfArgumentParser((ModelArguments, Arguments))
    model_args, args = parser.parse_args_into_dataclasses() 

    # save args to disk 
    if args.local_rank <= 0:
        merged_args = {**asdict(model_args), **asdict(args)}
        out_dir = deepcopy(args.output_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        #print("out_dir: ", out_dir, args.output_dir)
        with open(os.path.join(out_dir, "args.json"), "w") as f:
            ujson.dump(merged_args, f, indent=4)

    assert args.model_type in MODEL_TYPES

    # start building train dataset 
    if args.model_type in {"bert_splade", "t5_splade"}:
        if args.loss_type == "margin_mse":
            train_dataset = MarginMSEDataset(example_path=args.teacher_score_path,
                                             document_dir=args.collection_path,
                                             query_dir=args.queries_path)
            if args.model_type == "bert_splade":
                train_collator = MarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                                max_length=args.max_length)
            elif args.model_type == "t5_splade":
                train_collator = T5SpladeMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                                max_length=args.max_length)
            assert ("t5" in args.model_type and "t5" in model_args.model_name_or_path) or \
                    ("bert" in args.model_type and "bert" in model_args.model_name_or_path)
        else:
            raise NotImplementedError
    elif args.model_type in {"t5_dense"}:
        if args.loss_type == "margin_mse":
            train_dataset = MarginMSEDataset(example_path=args.teacher_score_path,
                                             document_dir=args.collection_path,
                                             query_dir=args.queries_path)
            train_collator = T5DenseMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                                max_length=args.max_length)
        else:
            raise NotImplementedError
    elif args.model_type in {"bert_term_encoder", "t5_term_encoder"}:
        if args.loss_type == "margin_mse":
            train_dataset = TermEncoderForMarginMSEDataset(example_path=args.teacher_score_path,
                                                           query_dir=args.queries_path,
                                                           docid_to_smtid_path=args.docid_to_smtid_path)
            if args.model_type == "t5_term_encoder":
                train_collator = T5TermEncoderForMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                               max_length=args.max_length)
            else:
                train_collator = TermEncoderForMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                               max_length=args.max_length)
            assert ("t5" in args.model_type and "t5" in model_args.model_name_or_path) or \
                    ("bert" in args.model_type and "bert" in model_args.model_name_or_path)
    elif args.model_type == "ripor":
        if args.loss_type in {"margin_mse", "direct_lng_knp_margin_mse"}: 
            train_dataset = RiporForMarginMSEDataset(dataset_path=args.teacher_score_path,
                                                    document_dir=args.collection_path,
                                                    query_dir=args.queries_path,
                                                    docid_to_smtid_path=args.docid_to_smtid_path,
                                                    smtid_as_docid=args.smtid_as_docid)
            train_collator = RiporForMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                                    max_length=args.max_length)
                                                    #args.pretrained_path, max_length=args.max_length)
        elif args.loss_type == "seq2seq":
            train_dataset = RiporForSeq2seqDataset(example_path=args.query_to_docid_path,
                                                    docid_to_smtid_path=args.docid_to_smtid_path)
            train_collator = RiporForSeq2seqCollator(tokenizer_type=model_args.model_name_or_path,
                                                    max_length=args.max_length)
        elif args.loss_type == "lng_knp_margin_mse":
            train_dataset = RiporForLngKnpMarginMSEDataset(dataset_path=args.teacher_score_path,
                                                    document_dir=args.collection_path,
                                                    query_dir=args.queries_path,
                                                    docid_to_smtid_path=args.docid_to_smtid_path,
                                                    smtid_as_docid=args.smtid_as_docid)
            train_collator = RiporForLngKnpMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                                    max_length=args.max_length)
        else:
            raise NotImplementedError
    elif args.model_type == "lexical_ripor":
        if args.loss_type in {"margin_mse", "direct_lng_knp_margin_mse"}: 
            train_dataset = LexicalRiporForMarginMSEDataset(dataset_path=args.teacher_score_path,
                                                    document_dir=args.collection_path,
                                                    query_dir=args.queries_path,
                                                    smt_docid_to_smtid_path=args.smt_docid_to_smtid_path,
                                                    lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
            train_collator = LexicalRiporForMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                                    max_length=args.max_length)
        elif args.loss_type == "kl_div":
            train_dataset = LexicalRiporForKLDivDataset(dataset_path=args.teacher_score_path,
                                                    document_dir=args.collection_path,
                                                    query_dir=args.queries_path,
                                                    smt_docid_to_smtid_path=args.smt_docid_to_smtid_path,
                                                    lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
            train_collator = LexicalRiporForKLDivCollator(tokenizer_type=model_args.model_name_or_path,
                                                    max_length=args.max_length)
        elif args.loss_type == "dense_pretrained_margin_mse":
            train_dataset = LexicalRiporForDensePretrainedMarginMSEDataset(dataset_path=args.teacher_score_path,
                                                                           document_dir=args.collection_path,
                                                                           query_dir=args.queries_path,
                                                                           lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
            train_collator = LexicalRiporForDensePretrainedMarginMSECollator(tokenizer_type=model_args.model_name_or_path,
                                                    max_length=args.max_length)
        elif args.loss_type == "seq2seq":
            train_dataset = LexicalRiporForSeq2seqDataset(example_path=args.query_to_docid_path,
                                                                       smt_docid_to_smtid_path=args.smt_docid_to_smtid_path, 
                                                                       lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
            train_collator = LexicalRiporForSeq2seqCollator(tokenizer_type=model_args.model_name_or_path,
                                                    max_length=args.max_length,
                                                    apply_lex_loss=args.apply_lex_loss)
        else:  
            raise NotImplementedError
    else:
        raise NotImplementedError

    # start initializing model 
    model_cls = model_cls_dict[args.model_type][args.loss_type] 
    if args.model_type in {"t5_splade", "t5_dense", "t5_term_encoder", "ripor", "lexical_ripor"}:
        model_args = None
    elif args.model_type in {"bert_splade", "bert_term_encoder"}:
        model_args = None
    else:
        raise NotImplementedError

    if args.model_type == "lexical_ripor" and args.loss_type == "seq2seq":
        model = model_cls.from_pretrained(args.pretrained_path, model_args, args.apply_lex_loss)
        if not args.apply_lex_loss:
            args.task_names.remove("lexical_rank")
            args.ln_to_weight.pop("lexical_rank")
    else:
        model = model_cls.from_pretrained(args.pretrained_path, model_args)

    # we should check the tokenizer is compatible with model for Ripor model
    if args.model_type == "ripor":
        if is_first_worker():
            print("the embedding size for ripor is {}".format(len(train_collator.tokenizer)))


    # start building evaluator 
    if args.model_type in {"bert_splade", "t5_splade"}:
        evaluator = None #initialize_splade_evaluator(model, args, model_args)
    else:
        evaluator = None
    

    # start building trainer 
    if True:
        if args.model_type in {"bert_splade", "t5_splade"}:
            training_args = Splade_TrainingArgs(
                output_dir=args.output_dir,
                do_train=True,
                do_eval=args.do_eval,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
                per_device_train_batch_size=args.per_device_train_batch_size,
                logging_steps=args.logging_steps,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps,
                disable_tqdm=False,
                load_best_model_at_end=False,
                dataloader_pin_memory=False,
                save_total_limit=5,
                seed=2,
                remove_unused_columns=False,
                task_names=args.task_names,
                ln_to_weight=args.ln_to_weight,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                full_rank_eval_qrel_path=args.full_rank_eval_qrel_path,
                bf16=args.use_fp16)
            
            assert args.max_steps >= 100, args.max_steps
            reg_to_reg_scheduler = {"doc_reg": RegWeightScheduler(lambda_=args.ln_to_weight["doc_reg"], T=training_args.max_steps // 3),
                                    "query_reg": RegWeightScheduler(lambda_=args.ln_to_weight["query_reg"], T=training_args.max_steps // 3)}
        elif args.model_type in {"bert_term_encoder", "t5_term_encoder", "t5_dense", "ripor", "lexical_ripor"}:
            training_args = TermEncoder_TrainingArgs(
                output_dir=args.output_dir,
                do_train=True,
                do_eval=args.do_eval,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
                per_device_train_batch_size=args.per_device_train_batch_size,
                logging_steps=args.logging_steps,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps,
                disable_tqdm=False,
                load_best_model_at_end=False,
                dataloader_pin_memory=False,
                save_total_limit=5,
                seed=2,
                remove_unused_columns=False,
                task_names=args.task_names,
                ln_to_weight=args.ln_to_weight,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                full_rank_eval_qrel_path=args.full_rank_eval_qrel_path,
                bf16=args.use_fp16)
            reg_to_reg_scheduler = None
        else:
            raise NotImplementedError
        
        trainer = SpladeTrainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=train_collator,
            args=training_args,
            reg_to_reg_scheduler=reg_to_reg_scheduler,
            splade_evaluator=evaluator
            )
    else:
        raise NotImplementedError
    
    
    if training_args.local_rank <= 0:  # only on main process
        wandb.login()
        wandb.init(project=args.wandb_project_name, name=args.run_name)
    
    # Let's save the tokenizer first 
    if is_first_worker():
        train_collator.tokenizer.save_pretrained(trainer.args.output_dir)
    trainer.train()
    trainer.save_torch_model_and_tokenizer(train_collator.tokenizer)

if __name__ == "__main__":
    main()




