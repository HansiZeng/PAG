# GD-RIPOR
This repo provides the source code and checkpoints for our paper [GD-RIPOR]()

## Package installation
- pip install -r requirements.txt 
- pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
- conda install -c conda-forge faiss-gpu 

## Download Files

## Inference 
We use a single 80GB A100 to run the script. Feel free to use other types of GPUs, such as V100, but it would be slower.
Make sure that the `task` variable in line 1 of `full_scripts/full_ripor_evaluate.sh` is set to `lexical_constrained_retrieve_and_rerank`, then run:
```
bash full_scripts/full_ripor_evaluate.sh
```


## Training 
All experiments are conducted 8x 40GB A100 GPUs. The whole training pipeline contains three stages: (1) Generative retrieval (GR) model  for set-based DocIDs. (2) GR model for sequence-based DocIDs. (3) Unified GR model for set-based & sequence-based DocIDs. Stages (1) and (2) can be train in parallel. 

### Stage 1: GR model for set-based DocIDs
The stage contains 2 phases: pre-training and fine-tunining. For pre-training, we train the GR model as a sparse encoder, then we select the top U words from the sparse vector for each document $d$ to form the set-based DocID, and we term it as $\{w^d_1, \ldots w^d_U \}$. For fine-tuning phase, we train the GR model for set-based DocID prediction.

#### Pre-training:
Run script for training the sparse encoder: 
```
bash full_scripts/t5_splade_train.sh 
```
Once model trained, run the following script to get the set-based DocIDs:
```
bash full_scripts/t5_splade_get_bow_rep.sh
```

#### Fine-tuning:
The apply the two-step fine-tuning stragegy. The negatives for the step 1 is from BM25, and the negatives for the step 2 is from the step 1 model itself.
For step 1 training, please first set `finetune_step=bm25_neg` in file `full_scripts/t5_full_term_encoder_train.sh`, then run:
```
full_scripts/t5_full_term_encoder_train.sh
```
For step 2 training, please first set `finetune_step=self_neg` in file `full_scripts/t5_full_term_encoder_train.sh`, then run:
```
full_scripts/t5_full_term_encoder_train.sh
```

### Stage 2: GR model for sequence-basded DocIDs
The stage also contains pre-training and fine-tuning phases. And the training pipline is the same as RIPOR [https://arxiv.org/pdf/2311.09134.pdf] except we don't use progressive training (We found the progressive training requires too much training time, and do not significantly influcen the model effectiveness).

#### pre-training 
We treat the GR mdoel as a dense encoder and apply the two-step training strategy:

For step 1,  please first set `finetune_step=bm25_neg` in file `full_scripts/t5_full_dense_train.sh`, then run:
```
bash full_scripts/t5_full_dense_train.sh
```
For step 2,  please first set `finetune_step=self_neg` in file `full_scripts/t5_full_dense_train.sh`, then run:
```
bash full_scripts/t5_full_dense_train.sh
```

After training we can run the following script to obtain sequence-based DocIDs. please first set `task=all_aq_pipline` in file `full_scripts/t5_full_dense_evaluate.sh`, then run:
```
bash full_scripts/t5_full_dense_evaluate.sh 
``` 
As shown in the paper, seq2seq pre-training can improve the GR model performance. Run the following script for training:
```
bash full_scripts/full_ripor_initial_train.sh
```

#### fine-tuning 
Let us apply the rank-oriented fine-tuning in this stage. Please run the script:
```
bash full_scripts/full_ripor_direct_lng_knp_train.sh
```

### Stage 3: Unified GR model for set-based & sequence-based DocIDs
We need to the merge weights of the above two trained GR models. 

First, move your terminal current directory to `t5_pretrainer`:
```
cd t5_pretrainer
```

Second, run the following code to merge weiths:
```
python -m full_preprocess.merge_model_weights
```

Third, move your terminal current directory back:
```
cd ..
```

Now, we can finally fine-tune the model. Run the following script:
```
bash full_scripts/full_lexical_ripor_train.sh
```


