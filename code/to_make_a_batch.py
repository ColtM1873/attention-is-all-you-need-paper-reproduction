import torch.nn as nn
import torch
from to_make_a_model import subsequent_mask
from torch.utils.data import Dataset, Sampler
import random
from array import array

class Batch:

    def __init__ ( self , src , tgt = None , pad = 2 ): # padding seq of tokens into the same
        # length in one batch, with lut row index as 2
        self.src = src # src has shape (nbatches , seq_len) with seq_len already uniformed
        self.src_mask = (src != pad).unsqueeze(-2) # into shape (nbatches , 1 , seq_len)
        # into a mask like ( T T T T T F F F )
        if tgt is not None:
            self.tgt = tgt [: , : -1] # leaving the last colume out
            self.tgt_y = tgt [ : , 1: ] # leaving the first colume out
            self.tgt_mask = self.make_std_mask (self.tgt  , pad)
            self.ntokens = (self.tgt_y != pad ).detach().sum().item()
            # with <s> uncounted, valid tokens number in this nbatches (n-sentences)

    @staticmethod
    def make_std_mask (tgt , pad): 
        " to hide padding and future tokens"
        tgt_mask = ( tgt != pad ) . unsqueeze (-2) # there is no future mask for src
        # compared to tgt
        # tgt being of shape ( nbatches , seq_len )
        # just like self.src_mask = (src != pad).unsqueeze(-2) 
        tgt_mask = tgt_mask & subsequent_mask(tgt.size (-1)) . type_as ( tgt_mask.data )
        # subsequent_mask returns shape (1, seq_len , seq_len)
        # operates with tgt_mask of ( nbatches , 1 , seq_len )
        # broadcast into shape (nbatches , seq_len , seq_len)
        return tgt_mask

class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, sp_model,  global_bos:int , global_eos:int, max_len=512, indices=None,):
        # indices is used to split into training and validation

        self.sp = sp_model
        self.src_lens = []
        self.tgt_lens = []
        self.src = []
        self.tgt = []
        indices_set = set(indices) if indices is not None else None
        with open(src_path , encoding = 'utf-8') as f_src, \
            open(tgt_path , encoding = 'utf-8') as f_tgt:
            for line_no, (s_line, t_line) in enumerate(zip(f_src, f_tgt)):
                if indices_set is not None and line_no not in indices_set:
                    continue
                s_ids = self.sp.encode(s_line.strip(), out_type=int)
                t_ids = self.sp.encode(t_line.strip(), out_type=int)
                s_ids.insert(0,global_bos)
                s_ids.append(global_eos)
                t_ids.insert(0,global_bos)
                t_ids.append(global_eos)
                
                if len(s_ids) <= max_len and len(t_ids) <= max_len:
                    self.src.append(array('I', s_ids))      # uint32 数组
                    self.tgt.append(array('I', t_ids))
                    self.src_lens.append(len(s_ids))        # 供 Sampler 用
                    self.tgt_lens.append(len(t_ids))
                    
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return torch.tensor(self.src[idx] , dtype = torch.long),\
            torch.tensor(self.tgt[idx] , dtype = torch . long)
    # return one dim tensor of shape (seq_len)

import random
from torch.utils.data import Sampler


class BucketingSampler(Sampler):
    """
    每个 epoch 随机打乱全部索引 → 按长度排序 → 动态分桶，
    保证：
      1) 桶内样本长度相近（减少 padding）
      2) 不同 epoch 的句子搭配完全不同
    """

    def __init__(
        self,
        dataset,                # TranslationDataset 实例
        tokens_per_batch: int,  # 每个 batch 允许的最大 token 数（含 pad）
        shuffle: bool = True,
        per_bucket_scale: float = 1.5,
    ):
        super().__init__()
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle
        self.per_bucket_scale = per_bucket_scale
        self.src_lens = dataset.src_lens        # list[int]，每个样本的源端长度
        self.indices = list(range(len(dataset)))

        # 初始化时建一次桶，仅用于 DataLoader 的 __len__（进度条）
        self.buckets = self._build_buckets(self.indices)
        # 记录一个估计的桶数（实际迭代中可能稍有变化，但对进度条足够）
        self._estimated_num_buckets = len(self.buckets)

    # ------------------------------------------------------------------
    #  建桶逻辑（不依赖 self，可复用）
    # ------------------------------------------------------------------
    def _build_buckets(self, indices):
        """
        按照长度排序后的 indices 建立桶列表。
        返回：list[list[int]]，每个内层列表是一个 batch 的样本索引。
        """
        # 按 src 长度排序（升序）
        indices = sorted(indices, key=lambda i: self.src_lens[i])

        buckets = []
        current_bucket = []
        current_max_len = 0

        for idx in indices:
            sample_len = self.src_lens[idx]

            if len(current_bucket) == 0:
                current_max_len = sample_len

            # 封桶条件一：当前桶已满（token 数达到上限）
            if len(current_bucket) >= self.tokens_per_batch / current_max_len:
                buckets.append(current_bucket)
                current_bucket = []
                current_max_len = sample_len

            # 封桶条件二：来了一个长度 > 当前最大长度 × scale 的句子
            elif (
                len(current_bucket) > 0
                and sample_len > current_max_len * self.per_bucket_scale
            ):
                buckets.append(current_bucket)
                current_bucket = []
                current_max_len = sample_len

            current_bucket.append(idx)

        # 最后一个桶
        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    # ------------------------------------------------------------------
    #  每个 epoch 的迭代入口
    # ------------------------------------------------------------------
    def __iter__(self):
        # 1. 获取全部索引（如果需要打乱）
        if self.shuffle:
            indices = self.indices.copy()
            random.shuffle(indices)            # 先打乱，以便相同长度的句子随机分组
        else:
            indices = self.indices             # 固定顺序（一般只用于验证集）

        # 2. 按当前 epoch 的索引重建桶
        buckets = self._build_buckets(indices)

        # 3. 打乱桶的顺序（但不打乱桶内顺序，桶内保持长度升序以最大化效率）
        if self.shuffle:
            random.shuffle(buckets)

        # 4. 更新以便 __len__ 仍然能用（非必须，但友好）
        self.buckets = buckets

        # 5. 逐 batch 返回
        for bucket in buckets:
            yield bucket

    # ------------------------------------------------------------------
    #  DataLoader 获取 batch 数量
    # ------------------------------------------------------------------
    def __len__(self):
        # 返回最近一次建桶的桶数（初始化时已有估计值）
        return len(self.buckets) if self.buckets else self._estimated_num_buckets


def collate_fn ( batch , pad_idx  ):
    """
    batch: a list of tuple in the form of ( src_tensor , tgt_tensor )
        name src_tensor in the shape of (src_len) , tgt_tensor in (tgt_len)
    """
    src_batch , tgt_batch = zip (*batch) # seperate src and tgt
    # that src_batch will be a tuple of src_tensor
    src_batch = nn.utils.rnn.pad_sequence(
            src_batch , batch_first = True , padding_value = pad_idx
            )
    # turn a tuple of variant length tensors into a 2 dim tensor
    # which unifies length, which will require padding
    tgt_batch = nn.utils.rnn.pad_sequence(
            tgt_batch , batch_first = True , padding_value = pad_idx
            )
    return src_batch , tgt_batch
    # which is of shape (nbatches , seq_len)


