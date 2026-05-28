import torch
import os
import time
import random
import sentencepiece
import os
import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import torch.nn.functional as F

import to_make_a_model
import to_make_a_batch
import to_make_a_train
import to_make_a_beam_test 
import save_and_print_parameters

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False
COMPILE = False

#---------------------------------------LoadSPMAndDataset---------------------------------------------
S_P_MODEL_PATH = "~/Desktop/EngChoVocabReallyGood.model"
OUT_DIR = "~/Desktop/preprocessed"

if DEBUG:
    SRC_PATH = "~/Desktop/train.debug.en" # initial parallel sentences in readable text
    TGT_PATH = "~/Desktop/train.debug.zh"
else:
    SRC_PATH = "~/Desktop/train.clean.en" # initial parallel sentences in readable text
    TGT_PATH = "~/Desktop/train.clean.zh"

#---------------------------------------ModelSaveAndLoad---------------------------------------------
CHECKPOINT_DIR = "~/Desktop/BestModel"
NUM_OF_COPIES = 3
VAL_MODEL_NAME = 'validate_best_model'
INF_MODEL_NAME = 'infer_print_model'
MODEL_SUFFIX = '.pt'
CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_DIR, (
    VAL_MODEL_NAME + str(0) + MODEL_SUFFIX
    ))

VAL_PARA_WRITE_IN_SHEET = "snapshot_after_val.txt"
INF_PARA_WRITE_IN_SHEET = "snapshot_after_inf.txt"
LOD_PARA_WRITE_IN_SHEET = "snapshot_after_load.txt"


BATCH_AGGREGATE_FACTOR = 8
BATCH_SCALE_FACTOR = 3
NOAM_FACTOR = 0.4
LABEL_SMOOTHING = 0.1
NAUGHTY_MODEL = 20000
NUM_EPOCHES = 400 
WARMUP_STEP = 10000 
VALIDATION_SET_SIZE = 50000 
MAX_TOKENS_PER_BUCKET = 1024 + 512 
VALIDATE_INTERVAL = 10000 * 2 
PRINT_INTERVAL = 2000 
D_MODEL = 1024
OPTIMIZER_LEARNING_RATE = 0.5
DROP_OUT = 0.1
if DEBUG:
    NOAM_FACTOR = 0.1
    LABEL_SMOOTHING = 0
    NAUGHTY_MODEL =500
    NUM_EPOCHES = 10000
    WARMUP_STEP = 100
    VALIDATION_SET_SIZE = 100
    MAX_TOKENS_PER_BUCKET = 2560
    VALIDATE_INTERVAL = 500
    PRINT_INTERVAL = 25
    D_MODEL = 1024
    OPTIMIZER_LEARNING_RATE = 0.2
    DROP_OUT = 0

INFER_INTERVAL = 0.5
PATIENCE = 20
PER_BUCKET_SCALE = 1.3
NUM_OF_HEADS = 16
NUM_WORKERS = 1 
# num of multiple threads allowed
BETA_ONE_ORI = 0.9
BETA_TWO_ORI = 0.98
BETA_ONE = BETA_ONE_ORI ** (1/BATCH_SCALE_FACTOR)
BETA_TWO = BETA_TWO_ORI ** (1/BATCH_SCALE_FACTOR)
RANDOM_SEED = 256
MAX_LEN = 128

#---------------------------------------SPMConstruct---------------------------------------------
sp_model = sentencepiece.SentencePieceProcessor()
sp_model.Load(S_P_MODEL_PATH)
global_pad_id  = sp_model.pad_id()
global_bos = sp_model.bos_id()
global_eos = sp_model.eos_id()

global_start_time = time.time()
print("Program initiated at :")
print(datetime.datetime.now())

def main():
    print(f"使用设备: {DEVICE}")
    os.makedirs( CHECKPOINT_DIR , exist_ok = True )
    vocab = sp_model.vocab_size()

#---------------------------------------MakeModel---------------------------------------------
    transformer_model = to_make_a_model.make_model(vocab , vocab , N =6 , d_model = D_MODEL , d_ff = D_MODEL * 4 , h = NUM_OF_HEADS , dropout= DROP_OUT)
    transformer_model.to(DEVICE) 

    print("Step one: MakeModel complete.")

#---------------------------------------SeperateTrainAndValidate---------------------------------------------
    with open(SRC_PATH) as f:
        num_of_lines = sum(1 for _ in f)
    random.seed (RANDOM_SEED)
    all_indices = list( range(num_of_lines) ) 
    random.shuffle(all_indices)
    val_indices = set (all_indices [ : VALIDATION_SET_SIZE ])
    train_indices = [ i for i in all_indices
            if i not in val_indices]
    print(f"总行数: {num_of_lines}")
    print(f"训练集: {len(train_indices)} 对")
    print(f"验证集: {len(val_indices)} 对")

#---------------------------------------TrainDataLoader---------------------------------------------
    full_train_data_set = to_make_a_batch.TranslationDataset(
            SRC_PATH , TGT_PATH , sp_model , 
            global_bos = global_bos , global_eos = global_eos,
            indices = train_indices,
            )
    print("full_train_data_set is set.")
    print(f"Time has passed {time.time() - global_start_time } seconds.")

    bucket_sampler = to_make_a_batch . BucketingSampler(
            full_train_data_set , MAX_TOKENS_PER_BUCKET , per_bucket_scale = PER_BUCKET_SCALE
            )
    print('bucket_sampler is set')
    dataloader = DataLoader (
            full_train_data_set , 
            batch_sampler = bucket_sampler , 
            collate_fn = partial(to_make_a_batch.collate_fn , pad_idx  = global_pad_id),
            num_workers = NUM_WORKERS , 
            pin_memory = True # True if GPU is used
            )
    print("Step two: TrainDataLoader complete.")
    print(f"Training dataset contains {full_train_data_set.__len__()} sentences.")

#---------------------------------------ValidateDataLoader---------------------------------------------
    full_validate_data_set = to_make_a_batch.TranslationDataset(
            SRC_PATH , TGT_PATH , sp_model  , 
            global_bos = global_bos , global_eos = global_eos,
            indices = list(val_indices),
            )
    bucket_sampler_val = to_make_a_batch . BucketingSampler(
            full_validate_data_set , MAX_TOKENS_PER_BUCKET, per_bucket_scale = PER_BUCKET_SCALE,
            shuffle = False
            )
    dataloader_val = DataLoader (
            full_validate_data_set , 
            batch_sampler = bucket_sampler_val , 
            collate_fn = partial(to_make_a_batch.collate_fn , pad_idx  = global_pad_id) ,
            num_workers = NUM_WORKERS , 
            pin_memory = True # True if GPU is used
            )

    print("Step three: ValidateDataLoader complete.")
    print(f"Training dataset contains {full_validate_data_set.__len__()} sentences.")

#---------------------------------------LearningRateAndOptimizer---------------------------------------------
    label_smoothing = to_make_a_train.LabelSmoothing( vocab , global_pad_id , LABEL_SMOOTHING)
    loss_compute = to_make_a_train.LossCompute( transformer_model.generator , label_smoothing )
    model_optimizer = torch . optim . Adam(
            transformer_model.parameters(),
            lr = OPTIMIZER_LEARNING_RATE,
            betas = (BETA_ONE , BETA_TWO),
            eps = 1e-9,
            )
    lr_scheduler = LambdaLR (
            optimizer = model_optimizer,
            lr_lambda = lambda step : to_make_a_train.rate(
                step , model_size = D_MODEL  , factor = NOAM_FACTOR ,  warmup = WARMUP_STEP
                )
            )

    print("Step Four: LearningRateAndOptimizer complete.")


#---------------------------------------initStatus---------------------------------------------
    best_validation_loss = float('inf')
    no_improve_valid = 0
    global_step = 0
    print_gap = 0
    start = time.time()
    transformer_model.train()
    aggregate_total_loss_for_batch = 0
    aggregate_tgt_tokens_volumn_per_print_interval = 0

    print("Step Five: initStatus complete.")
#---------------------------------------LoadSavedModel---------------------------------------------
    if os.path.isfile(CHECKPOINT_FILE_PATH):
        checkpoint = torch.load(CHECKPOINT_FILE_PATH, map_location=DEVICE)
        cleaned_state_dict = {
            k.replace('_orig_mod.', ''): v 
            for k, v in checkpoint['model_state_dict'].items()
        }
        transformer_model.load_state_dict(cleaned_state_dict)        
        model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['step']
        best_validation_loss = checkpoint['val_loss']
        lr_scheduler.load_state_dict (checkpoint['scheduler_state_dict'])
        print("Load model successfully")
        print(f"with global_step being{global_step}")
        print(f"best_validation_loss being {best_validation_loss}")
        print("_" * 50)

    print("Step Six: LoadSavedModel complete.")

#---------------------------------------PrintModelPara---------------------------------------------
    save_and_print_parameters.print_and_write_para(
            model = transformer_model ,
            optimizer = model_optimizer ,
            prefix = "After Loading" ,
            filepath = LOD_PARA_WRITE_IN_SHEET
            )

#---------------------------------------PrintTime---------------------------------------------
    print(f"Time already used: { float(time.time() - global_start_time) :.1f} seconds.")
    print(f"Total num of sentence-pairs used for training: {len(full_train_data_set)} . ")
    print(f"Total num of batches each epoch: {len(bucket_sampler)} .")
    print ("Start training transformer_model.")
    print("#"*50)
    print('\n')

#---------------------------------------StartTraining---------------------------------------------
#---------------------------------------StartTraining---------------------------------------------
#---------------------------------------StartTraining---------------------------------------------
#---------------------------------------StartTraining---------------------------------------------
#---------------------------------------StartTraining---------------------------------------------
#---------------------------------------StartTraining---------------------------------------------
#---------------------------------------StartTraining---------------------------------------------
    if COMPILE:
        transformer_model = torch.compile(transformer_model, dynamic=True)
    epoch_start = time.time()
    #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    if True:
        for epoch in range(NUM_EPOCHES):
            transformer_model.train()
            epoch_elapse = time.time() - epoch_start
            epoch_start = time.time()
            OOM_NUM = 0
            for step , (src_batch , tgt_batch ) in enumerate (dataloader):
                try:

#---------------------------------------PrintBatch---------------------------------------------
                    if not DEBUG:
                        if not (step % NAUGHTY_MODEL):
                            print(src_batch[0])
                            str_show = sp_model.decode(src_batch[0].detach().tolist())
                            print(f"The src str being:\n{str_show}")
                            print(tgt_batch[0])
                            str_show = sp_model.decode(tgt_batch[0].detach().tolist())
                            print(f"The tgt str being:\n{str_show}")
                            print(f"Current step being {step}")
                    global_step += 1

#---------------------------------------MakeBatch---------------------------------------------
                    current_batch = to_make_a_batch.Batch(src_batch , tgt_batch , global_pad_id)
                    src       = current_batch.src.to(DEVICE)
                    tgt       = current_batch.tgt.to(DEVICE)
                    src_mask  = current_batch.src_mask.to(DEVICE)
                    tgt_mask  = current_batch.tgt_mask.to(DEVICE)
                    tgt_y     = current_batch.tgt_y.to(DEVICE)
                    ntokens   = current_batch.ntokens

#---------------------------------------ForwardAndBack---------------------------------------------
                    output_of_model = transformer_model(
                        src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask
                    )
                            # src and tgt here is of shape (nbatches , seq_len)
                             # the model only returns a (nbatches , seq_len , d_model) shape tensor
                    total_loss_for_batch , avg_loss_as_scalar = loss_compute(
                            output_of_model , tgt_y , ntokens
                            )
                    avg_loss_as_scalar = avg_loss_as_scalar / BATCH_AGGREGATE_FACTOR
                    avg_loss_as_scalar.backward()
                    #torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
                    if global_step % BATCH_AGGREGATE_FACTOR == 0:
                        # step starts from zero, thus can`t be used here
                        # and global_step is the perfect one here
                        # require VALIDATE_INTERVAL and VALIDATE_INTERVAL * INFER_INTERVAL is the integer multiple of BATCH_AGGREGATE_FACTOR
                        model_optimizer.step()
                        model_optimizer.zero_grad( set_to_none = True )
                        lr_scheduler . step()

#---------------------------------------PrintPrintInterval---------------------------------------------
                    aggregate_tgt_tokens_volumn_per_print_interval += ntokens
                    aggregate_total_loss_for_batch += float(
                        total_loss_for_batch.item()
                        )
                    print_gap += 1
                    if(print_gap >= PRINT_INTERVAL):
                        avg_loss_per_token =(
                             aggregate_total_loss_for_batch /
                             aggregate_tgt_tokens_volumn_per_print_interval
                                )
                        elapsed = time.time() - start
                        lr = model_optimizer.param_groups[0]["lr"]
                        print( "\n" , "#" * 50)
                        print(f"Epoch: {epoch} | Step no : {step + 1}  | Global_step : {global_step} . ")
                        print(f"The average avg_loss_as_scalar until last PRINT_INTERVAL is {avg_loss_per_token:.3f}.")
                        print(f"The last step takes {elapsed:.3f} secondes , and the current learning rate is {lr * 100000:.4f} /100000." )
                        print(f"And the program has been running for { float( time.time() - global_start_time):.1f} seconds" )
                        print("#" * 50 , "\n")
                        print_gap = 0
                        aggregate_total_loss_for_batch = 0
                        aggregate_tgt_tokens_volumn_per_print_interval = 0
                    start = time.time()

                    del total_loss_for_batch
                    del avg_loss_as_scalar

#---------------------------------------Validation---------------------------------------------
#---------------------------------------Validation---------------------------------------------
                    if( global_step % VALIDATE_INTERVAL == 0 ):
                        #torch.cuda.empty_cache()
                        transformer_model.eval()
                        current_val_loss_sum = 0.0
                        with torch.no_grad():
                            for step_val , (src_batch_val , tgt_batch_val ) in enumerate (dataloader_val):
                                current_batch_val = to_make_a_batch.Batch(src_batch_val , tgt_batch_val , global_pad_id)
                                src_val      = current_batch_val.src.to(DEVICE)
                                tgt_val      = current_batch_val.tgt.to(DEVICE)
                                src_mask_val = current_batch_val.src_mask.to(DEVICE)
                                tgt_mask_val = current_batch_val.tgt_mask.to(DEVICE)
                                tgt_y_val    = current_batch_val.tgt_y.to(DEVICE)
                                ntokens_val    = current_batch_val.ntokens

#---------------------------------------ForwardAndBack---------------------------------------------
                                output = transformer_model(src_val, tgt_val, src_mask_val, tgt_mask_val)
                                total_loss_for_batch_val, avg_loss_as_scalar_val = loss_compute(output, tgt_y_val, ntokens_val) 
                                # src and tgt here is of shape (nbatches , seq_len)
                                # the model only returns a (nbatches , seq_len , d_model) shape tensor
                                current_val_loss_sum += total_loss_for_batch_val.item()
                                del total_loss_for_batch_val
                                del avg_loss_as_scalar_val

#---------------------------------------PrintValidate---------------------------------------------
                            print("\n" ,"$-"*20)
                            print(f"validation complete! \n"
                                f"with current_val_loss_sum being {current_val_loss_sum} and best_validation_loss being {best_validation_loss}\n"
                                f"current patience( no_improve_valid) being {no_improve_valid} out of {PATIENCE}.\n")
                            print("And the time being:")
                            print(datetime.datetime.now())
                            print("$-"*20,"\n")

#---------------------------------------SaveModelInValidation---------------------------------------------
                            if current_val_loss_sum < best_validation_loss:
                                best_validation_loss = current_val_loss_sum
                                no_improve_valid = 0

                                save_and_print_parameters.save_and_print_and_write_para(
                                        model = transformer_model ,
                                        optimizer = model_optimizer ,
                                        prefix = "Validate Save" , 
                                        filepath = VAL_PARA_WRITE_IN_SHEET,
                                        num_of_copies = NUM_OF_COPIES,
                                        global_step = global_step,
                                        best_validation_loss = best_validation_loss,
                                        lr_scheduler = lr_scheduler,
                                        checkpoint_dir = CHECKPOINT_DIR,
                                        file_name = VAL_MODEL_NAME,
                                        file_suffix = MODEL_SUFFIX,
                                        prompt = "Model Saved from validate!"
                                        )
                            else :
                                no_improve_valid += 1
                                if no_improve_valid >= PATIENCE:
                                    return
                        #with torch.no_grad():
                        transformer_model.train()
                    #if( global_step % VALIDATE_INTERVAL == 0 ):


#---------------------------------------INFER_INTERVALInferAndSave---------------------------------------------
                    if ( global_step % (VALIDATE_INTERVAL * INFER_INTERVAL) == 0  ):

                        save_and_print_parameters.save_and_print_and_write_para(
                                model = transformer_model ,
                                optimizer = model_optimizer ,
                                prefix = "Inference Save" , 
                                filepath = INF_PARA_WRITE_IN_SHEET,
                                num_of_copies = NUM_OF_COPIES,
                                global_step = global_step,
                                best_validation_loss = best_validation_loss,
                                lr_scheduler = lr_scheduler,
                                checkpoint_dir = CHECKPOINT_DIR,
                                file_name = INF_MODEL_NAME,
                                file_suffix = MODEL_SUFFIX,
                                prompt = "Model Saved from infer!"
                                )
                                            
                        transformer_model.eval()
                        print("\n" ,"$-"*20)
                        #with torch.autocast(device_type='cuda', enabled = False):
                        if True:
                            to_make_a_beam_test.main_test(
                                    sp_model , transformer_model ,
                                    MAX_LEN , global_start_symbol = global_bos ,
                                    device_beam = DEVICE , global_pad_id = global_pad_id ,
                                    global_end_symbol = global_eos , vocab_size = vocab)
                        print("$-"*20,"\n")
                        transformer_model.train()

#---------------------------------------DealWithOOM---------------------------------------------
                #try:
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('_' * 50 )
                        print('CUDA OOM ,skip one batch.')
                        OOM_NUM += 1
                        print(f'CUDA OOM already happened {OOM_NUM} times in this` times` epoch.')
                        torch.cuda.empty_cache()
                        model_optimizer.zero_grad()
                        print('_' * 50 )
                        continue
                    else:
                        raise e
            #for step , (src_batch , tgt_batch ) in enumerate (dataloader):
        #for epoch in range(NUM_EPOCHES):
    #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#def main():


if __name__ == "__main__":
    main()



        

