import torch
import torch.nn as nn

BATCH_SCALE_FACTOR = 2

def rate (step , model_size , factor  , warmup):
    # model_size is d_model
    # step is current step( that is one batch )

    if step == 0:
        step =1
    return factor * (
            model_size ** (-0.5) * 
            min(
                (step/BATCH_SCALE_FACTOR) ** (-0.5) , 
                (step/BATCH_SCALE_FACTOR) * warmup ** (-1.5))
            )
    
class LabelSmoothing( nn.Module ):

    def __init__ (self , size , padding_idx , smoothing = 0.0):
        # here size means tgt_vocab size
        super().__init__()
        self.criterion = nn.KLDivLoss ( reduction = "sum" )
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward( self , x, target ):
        # x is the output of generator , of shape (seq_len , tgt_vocab)
        # x already after log_softmax
        # target is 1 dim of shape (seq_len)
        assert x.size(1) == self.size
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        assert target.min() >= 0 and target.max() < self.size, \
	    f"target out of range: min={target.min()}, max={target.max()}, vocab_size={self.size}"
        true_dist.scatter_ (1, target.detach().unsqueeze(1) , self.confidence)
        true_dist [ : , self.padding_idx ] = 0
        mask = torch . nonzero (target . detach() == self . padding_idx)
        if mask.dim() > 0 :
            true_dist.index_fill_(0, mask.squeeze() , 0.0)

        self.true_dist = true_dist
        return self.criterion (x , true_dist .clone() .detach())


class LossCompute:

    def __init__ (self , generator , criterion): # an instance of LabelSmoothing should be passed in as criterion
        self.generator = generator # and generator belongs to transformer_model
        self.criterion = criterion
    def __call__ (self  , x , y ,norm): # here, x is the output of transformer_model,
        # of shape (nbatches , seq_len , d_model)
        # y is tgt_y of Batch , of shape ( nbatches , seq_len ), whose elems are int index
        # And norm
        # norm here means the non-padding tokens` total num in this very batch
        x = self. generator(x) # turning x into (nbatches , seq_len , tgt_vocab)
        sloss = (
                self.criterion (
                    x.contiguous() . view(-1 , x.size(-1)) , y.contiguous() . view(-1)
                    # push x into (nbatches * seq_len , tgt_vocab) 
                    # push y into (nbatches * seq_len)
                    # And in LabelSmoothing, y will be unsqueeze(1)
                    )
                /norm # divide by norm
                # cause LabelSmoothing use sum mode, so /norm will stablize learning rate(grad , more precisely)
                )
        return sloss.detach().clone() * norm , sloss
        # multiply it back
        # the first return value is so called total loss



