import math
import torch
import sentencepiece
import os

import to_make_a_batch
import to_make_a_train
import to_make_a_model

BEAM_START = 5
BEAM_ADD = 1
BEAM_ADD_INVERVAL = 30
#EPSILON = 0.2
REPEAT_SCALE = 0.7
REDUCE_SCALE =  math.log(1/REPEAT_SCALE)


def beam_search_decode(model, src, src_mask, max_len,  device_beam , global_start_symbol , global_end_symbol , vocab):
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
            input_as_tgt = torch.tensor(batch_tokens, dtype=src.dtype, device=device_beam)
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

                for i in range (len(current_seq)):
                    repeated_id = current_seq[i]
                    current_log_prob[repeated_id] -= REDUCE_SCALE

                with_history_prob = [
                    pro + current_history_prob for pro in current_log_prob
                ]
                total_prob_list_comparable.extend(with_history_prob)

            total_prob_list_comparable_tensor = torch.tensor(
                total_prob_list_comparable, device=device_beam
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


def main_test( sp_model, transformer_model ,max_len , global_start_symbol , device_beam , global_pad_id , global_end_symbol ,vocab_size):

        eng_str = "I have nothing to offer but blood, toil, tears, and sweat. We have before us an ordeal of the most grievous kind. We have before us many, many months of struggle and suffering. You ask, what is our aim? I can answer in one word. It is victory. Victory at all costs - Victory in spite of all terrors - Victory, however long and hard the road may be, for without victory there is no survival."
        print(eng_str)
        src = torch.tensor(
            sp_model.encode(eng_str, out_type=int),
            dtype=torch.long
        ).unsqueeze(0).to(device_beam)

        batch_test = to_make_a_batch.Batch(src=src, pad=global_pad_id)

        return_gen = beam_search_decode(
            transformer_model, src, batch_test.src_mask, max_len, device_beam , global_end_symbol = global_end_symbol , global_start_symbol = global_start_symbol ,  vocab = vocab_size
        )

        out_sentence = sp_model.decode(return_gen)
        print("The translation is as below:\n\t", out_sentence)

