# attention-is-all-you-need-paper-reproduction
Paper reproduction for the original transformer paper.
## Introduciton
This repository stores codes for paper reproduction of the famous *Attention is all you need*. And guided by *The Annotated Transformer*. Yet the translation task is changed to translate English into Chinese.
## Reference
In code directory,  
almost all codes in to\_make\_a\_model.py and to\_make\_a\_train.py ,  
and part of codes in to\_make\_a\_batch.py and main.py  
are from *The Annotated Transformer*, wroten by Vaswani et al. in 2022, presented by Harvard NLP.  
Though codes remain the same, they are extensively annotated in my code files.  
## Structure
Code directory contains code files.  
de-bug-data directory contains two dataset for debugging. The whole scale training data can`t be uploaded here.
sp-model directory contains trained sentence piece model and vocab for my model training.
## Deploy Tech Choice
- Embedding weight sharing. 
- Beam search inference.
- Matching sample length bucketing.
- Aggregate gradients
- Early stop based upon KLDivergence
- Label Smoothing
- Make inference during training
## Hyper Parameters Explanation
### main.py
- DEBUG: print info more frequently, use much smaller dataset to quickly run through and debug the program.
- COMPILE: indicates pytorch`s computation graph compilation.  
    - But note that, when compiled, the inference test during training could output questionable string.
- S\_P\_MODEL\_PATH: the path where sentence piece model is located.
- SRC\_PATH: the path where source language dataset(lines of sentences) is located
- TGT\_PATH: as above
- CHECKPOINT\_DIR: the path where saved models should be placed.
- NUM\_OF\_COPIES: copies per save
- VAL\_MODEL\_NAME: the file name of model saved after validation
- INF\_MODEL\_NAME: the file name of model saved after inference
- MODEL\_SUFFIX: suffix of saved model file
- VAL\_PARA\_WRITE\_IN\_SHEET: log of model parameters` file name
- INF\_PARA\_WRITE\_IN\_SHEET: as above
- LOD\_PARA\_WRITE\_IN\_SHEET: write log of model parameters once model is loaded
- BATCH\_AGGREGATE\_FACTOR: to aggregate batches and apply gradients for once, for limited GPU memory to simulate large batch.
- BATCH\_SCALE\_FACTOR: to slow down learning rate change and prolong optimizer memory of time series. used along with NOAM\_FACTOR approximately 1 divided by BATCH\_SCALE\_FACTOR.
    - to apply the non-gradient-aggregate, but use smaller learning rate and slower learning rate change approach, for limited GPU memory
- NOAM\_FACTOR: Noam learning rate schedule factor.
- LABEL\_SMOOTHING: label\_smoothing
- NAUGHTY\_MODEL: steps interval to print some training sample. to illustrate possible sampling mistakes.
- NUM\_EPOCHES: epoches to run
- WARMUP\_STEP: warm up for Noam learning rate schedule
- VALIDATION\_SET\_SIZE: number of sentence pairs used for validation
- MAX\_TOKENS\_PER\_BUCKET: which is batch size measure by number of tokens, should suit GPU memory
- VALIDATE\_INTERVAL: the steps interval of validation to take place
- PRINT\_INTERVAL: the steps interval to print current training info
- D\_MODEL: the dimension of model
- OPTIMIZER\_LEARNING\_RATE: lr rate input into adam optimizer
- DROP\_OUT: drop out probability applied in model
- INFER\_INTERVAL: multiply VALIDATE\_INTERVAL to regulate the interval to execute inference during training
- PATIENCE: to early stop when validation loss stop dropping
- PER\_BUCKET\_SCALE: to regulate bucket diverse, samples in one bucket shouldn`t be out of this relative scale
- NUM\_OF\_HEADS: CPU used for the task, could boost memory usage if too high
- BETA\_ONE\_ORI: BETA\_ONE before revised with BATCH\_SCALE\_FACTOR
- RANDOM\_SEED: random seed
- MAX\_LEN: length of tokens per sample(sentence)
## Deploy Guidence
The current code is executable on transformer big model to produce quality-guaranted model.
One should adjust the hyper parameters: MAX\_TOKENS\_PER\_BUCKET, NOAM\_FACTOR, BATCH\_AGGREGATE\_FACTOR and BATCH\_SCALE\_FACTOR according to their GPU memory and other hardware specification.
##But be aware that there are three empirical suggestions to produce a heathy model:## 
- ##MAX\_TOKENS\_PER\_BUCKET multiplies BATCH\_AGGREGATE\_FACTOR should exceeds 10000##
- ##BATCH\_SCALE\_FACTOR mulplies NOAM\_FACTOR should approximately equals one##
- ##WARMUP should be within (8000,12000) scope##


