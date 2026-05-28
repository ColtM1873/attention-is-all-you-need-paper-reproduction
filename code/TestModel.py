import math
import torch
import sentencepiece
import os

import to_make_a_batch
import to_make_a_train
import to_make_a_model

DEBUG = 1
VALID_MODEL = 0
INFER_MODEL = 1 - VALID_MODEL

REPEAT_PENAL = math.log(0.9)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "/home/nill/Desktop/VC/BestModel"
S_P_MODEL_PATH = "/home/nill/Desktop/VC/God/EngChoVocabReallyGood.model"
MAX_LEN = 128

sp_model = sentencepiece.SentencePieceProcessor()
sp_model.Load(S_P_MODEL_PATH)
global_start_symbol = sp_model.bos_id()
global_end_symbol = sp_model.eos_id()
global_pad_id = sp_model.pad_id()
vocab = sp_model.vocab_size()
D_MODEL = 1024
HEADS = 16


transformer_model = to_make_a_model.make_model(vocab , vocab , N =6 , d_model = D_MODEL , d_ff = D_MODEL * 4 , h = HEADS )
transformer_model.to(DEVICE)

if VALID_MODEL:
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, 'validate_best_model1.pt'),
        map_location=("cuda" if torch.cuda.is_available() else "cpu")
    )
elif INFER_MODEL:
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, 'infer_print_model0.pt'),
        map_location=("cuda" if torch.cuda.is_available() else "cpu")
    )

cleaned_state_dict = {
    k.replace('_orig_mod.', ''): v 
    for k, v in checkpoint['model_state_dict'].items()
}
transformer_model.load_state_dict(cleaned_state_dict)
transformer_model.eval()

BEAM_START = 15
BEAM_ADD = 1
BEAM_ADD_INVERVAL = 30
EPSILON = 0.2

def inspect_model(model, prefix="", check_grad=False):
    """
    打印模型参数的关键信息：dtype, device, norm, std, min, max
    check_grad=True 时同时打印梯度信息
    """
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"[{prefix}] 模型参数诊断")
    print(f"{sep}")
    
    header = f"{'参数名':<55s} {'dtype':<15s} {'norm':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
    if check_grad:
        header += f" {'grad_norm':>10s} {'grad_mean':>10s}"
    print(header)
    print("-" * (len(header) + 10))
    
    for name, param in model.named_parameters():
        p = param.data
        dtype_str = str(p.dtype).replace("torch.", "")
        
        line = (
            f"{name:<55s} "
            f"{dtype_str:<15s} "
            f"{p.norm().item():10.4f} "
            f"{p.std().item():10.4f} "
            f"{p.min().item():10.4f} "
            f"{p.max().item():10.4f}"
        )
        
        if check_grad and param.grad is not None:
            g = param.grad
            line += f" {g.norm().item():10.4f} {g.mean().item():10.4f}"
        elif check_grad:
            line += f" {'None':>10s} {'None':>10s}"
        
        print(line)


    # 汇总
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"{sep}\n")

inspect_model(transformer_model)

def beam_search_decode(model, src, src_mask, max_len, start_symbol):
    # src: (1, src_seq_len)
    memory = model.encode(src, src_mask)
    # memory: (1, src_seq_len, d_model)

    still_generating_list = []
    already_end_list = []
    still_generating_list.append([[global_start_symbol], 0])
    current_tgt_length = len(still_generating_list[0][0])   # = 1

    with torch.no_grad():
        while True:
            generating_list_length = len(still_generating_list)

            # ---- 长度上限：强制全部结束 ----
            if current_tgt_length >= max_len - 1:
                for x in still_generating_list:
                    x[0].append(global_end_symbol)
                    x[1] = x[1] / current_tgt_length
                already_end_list = already_end_list + still_generating_list
                break

            if generating_list_length == 0:
                break

            current_beam_size = (
                BEAM_START
                + (current_tgt_length * BEAM_ADD) // BEAM_ADD_INVERVAL
                - len(already_end_list)
            )
            if current_beam_size <= 0:
                break

            # ============================================================
            #  batch decode：所有 beam 一次 forward 完成
            # ============================================================
            batch_tokens = [item[0] for item in still_generating_list]
            input_as_tgt = torch.tensor(batch_tokens, dtype=src.dtype, device=DEVICE)
            # shape: (num_beams, current_tgt_length)

            num_beams = input_as_tgt.size(0)

            # memory / src_mask 按 batch 维度展开
            memory_expanded = memory.expand(num_beams, -1, -1)
            src_mask_expanded = src_mask.expand(num_beams, -1, -1)
            tgt_mask = to_make_a_model.subsequent_mask(
                input_as_tgt.size(1)
            ).type_as(src)

            out = model.decode(
                memory_expanded, src_mask_expanded, input_as_tgt, tgt_mask
            )
            # out: (num_beams, current_tgt_length, d_model)

            log_probs_batch = model.generator(out[:, -1])
            # shape: (num_beams, vocab_size)
            log_probs_list = log_probs_batch.tolist()
            # list of lists, each inner list length = vocab

            total_prob_list_comparable = []

            for i in range(generating_list_length):
                current_seq = still_generating_list[i][0]
                current_history_prob = still_generating_list[i][1]
                current_log_prob = log_probs_list[i]
                for rid in current_seq:
                    current_log_prob[rid] += REPEAT_PENAL
                with_history_prob = [
                    pro + current_history_prob for pro in current_log_prob
                ]
                total_prob_list_comparable.extend(with_history_prob)

            total_prob_list_comparable_tensor = torch.tensor(
                total_prob_list_comparable, device=DEVICE
            )
            k = min(current_beam_size, len(total_prob_list_comparable))
            large_value_vec, large_index_vec = torch.topk(
                total_prob_list_comparable_tensor, k,
                dim=0, largest=True, sorted=True
            )

            new_generating_list = []
            for value, index in zip(
                large_value_vec.tolist(), large_index_vec.tolist()
            ):
                parent_id = index // vocab
                token_id = index % vocab

                new_seq = still_generating_list[parent_id][0] + [token_id]
                new_member = [new_seq, value]

                if token_id == global_end_symbol:
                    new_member[1] = new_member[1] / current_tgt_length
                    already_end_list.append(new_member)
                else:
                    new_generating_list.append(new_member)

            still_generating_list = new_generating_list
            current_tgt_length += 1
            if DEBUG:
                print("-" * 50)
                print(f"\nThis is the {current_tgt_length} iter. Candidates are as follow:")
                print("-" * 50)

                for iter_i in range(len(still_generating_list)):
                    list_i = still_generating_list[iter_i][0]
                    str_i = sp_model.decode(list_i)
                    print(str_i)
                    


    # ---- 兜底：万一 already_end_list 为空 ----
    if not already_end_list:
        if still_generating_list:
            already_end_list = still_generating_list
        else:
            # 极端情况：所有 beam 都丢了
            return [global_start_symbol, global_end_symbol]

    max_index = max(
        range(len(already_end_list)),
        key=lambda i: already_end_list[i][1]
    )
    return already_end_list[max_index][0]


def process_line(line : str ) -> str :
    to_list = sp_model.encode(line, out_type=int)
    to_list.insert(0,global_start_symbol)
    to_list.append(global_end_symbol)

    src = torch.tensor(
        to_list,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    batch_test = to_make_a_batch.Batch(src=src, pad=global_pad_id)

    return_gen = beam_search_decode(
        transformer_model, src, batch_test.src_mask, MAX_LEN, global_start_symbol
    )

    out_sentence = sp_model.decode(return_gen)
    return out_sentence.strip()

def process_file( input_file_path : str , output_file_path : str ) :
    with open( input_file_path , 'r' , encoding = 'utf-8' ) as f_in , \
            open (output_file_path , 'w' , encoding = 'utf-8' ) as f_out:
                for idx , line in  enumerate (f_in ,  start = 1):
                    processed = process_line(line)
                    f_out.write(processed +'\n')

                    if idx % 100 == 0:
                        print(f"Already processed {idx} lines.")

def old_main():
    while True:
        eng_str = input(
            " input your English sentence to translate into Chinese here\n \t => :"
        )

        to_list = sp_model.encode(eng_str, out_type=int)
        to_list.insert(0,global_start_symbol)
        to_list.append(global_end_symbol)

        src = torch.tensor(
            to_list,
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)

        batch_test = to_make_a_batch.Batch(src=src, pad=global_pad_id)

        return_gen = beam_search_decode(
            transformer_model, src, batch_test.src_mask, MAX_LEN, global_start_symbol
        )

        out_sentence = sp_model.decode(return_gen)
        print("The translation is as below:\n\t", out_sentence)


if __name__ == "__main__":
    #process_file( SRC_FILE_PATH , OUTPUT_FILE_PATH )
    old_main()
