import torch.nn.functional as F
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy
import time
import warnings


warnings.filterwarnings("ignore")


class EncoderDecoder(nn.Module):
    
    def __init__ (self , encoder , decoder , src_embed , tgt_embed , generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src , tgt , src_mask , tgt_mask ):
        return self.decode (
                self.encode( src , src_mask ),
                src_mask,
                tgt,
                tgt_mask,
                )

    def encode( self, src , src_mask ):
        return self.encoder( self.src_embed( src ) , src_mask,)

    def decode( self , memory , src_mask , tgt , tgt_mask ):
        return self.decoder (self.tgt_embed( tgt ) , memory , src_mask, tgt_mask)


class Generator( nn.Module ):
    """After one  trainable linear, change the last dim of tensor into log_softmax"""

    def __init__(self, embed_weight):
        super().__init__()
        self.embed_weight = embed_weight
        #self.pre_norm = nn.LayerNorm(embed_weight.size(1), eps = 1e-6)
        
    def forward( self, x ):
        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
        if True:
        #return log_softmax(F.linear(self.pre_norm(x), self.embed_weight), dim=-1)
            return log_softmax(F.linear(x, self.embed_weight), dim=-1)

def clones (module , N):
    return nn.ModuleList( [copy.deepcopy(module)  for  _ in range(N)] )#return a ModuleList 
# of seperately trainable layers


class Encoder( nn.Module ):
    """ To pass in the EncoderDecoder class as encoder """

    def __init__(self , layer , N):
        super().__init__()
        self.layers = clones (layer , N) # a ModuleList object
        self.norm = LayerNorm ( layer.size )

    def forward(self , x ,mask):
        for layer in self.layers:
            x = layer( x , mask ) # as the stacked layers, prior layer`s output being 
            # next layer`s input
        return self.norm(x) # after N layers( each layer contains two sublayers , 
    # one mutihead , one add and norm, one fully connect , one add and norm , in total)
    # Append another LayerNorm


class LayerNorm( nn.Module ):

    def __init__(self , features , eps = 1e-6): # features is d_model, an int
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)) # scaling factor
        self.beta = nn.Parameter(torch.zeros(features)) # shift factor
        self.eps = eps

    def forward(self,x):
        mean = x.mean( -1 , keepdim = True ) #The last dim, that is d_model dim, calc mean
        std = x.std( -1 , keepdim = True  )
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection( nn.Module ):

    def __init__(self , size , dropout):
        super().__init__()
        self.norm = LayerNorm( size ) # size is the size of the last dim, features or d_model
        # this size variant is for initializing the scaling factor and shift factor
        self.dropout = nn.Dropout( dropout ) # dropout being the probability of dropout

    def forward(self , x , sublayer):
        return x + self.dropout( sublayer (self.norm(x)) ) # the sublayer variant here will be 
    # passed in as a lambda function in the EncoderLayer defination, which will be formed by the
    # feed_forward or self_attn


class EncoderLayer (nn.Module):
    
    def __init__ ( self , size , self_attn , feed_forward , dropout ):# that is one layer, yet
        # the SublayerConnection will be applied to its two sublayers
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones ( SublayerConnection ( size , dropout ) , 2 ) #the sublayer here 
        # means two sublayers
        self.size = size

    def forward( self , x , mask ):
        x = self.sublayer[0] ( x , lambda x : self.self_attn (x,x,x ,mask) ) # Q, K ,V
        return self.sublayer[1] ( x, self.feed_forward )


class Decoder( nn.Module ):

    def __init__(self , layer , N):
        super().__init__()
        # same as Encoder , but different with forward
        self.layers = clones( layer , N )
        self.norm = LayerNorm( layer.size )

    def forward(self , x , memory , src_mask , tgt_mask): # with memory and tgt_mask added
        for layer in self.layers:
            x = layer( x , memory , src_mask , tgt_mask ) #except for x, each layer shares the 
            # same inputs
        return self.norm(x) # one extra LayerNorm as usual


class DecoderLayer( nn.Module ):

    def __init__( self , size , self_attn , src_attn, feed_forward , dropout ):
        super().__init__()
        self.size = size
        self.self_attn = self_attn # the multihead attn will be passed in
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones ( SublayerConnection( size , dropout ) ,3  )

    def forward (self, x , memory , src_mask , tgt_mask): # same as Decoder forward
        m = memory
        x = self.sublayer[0] ( x , lambda x : self.self_attn ( x,x,x,tgt_mask ) )
        # already generated seq( above )
        x = self.sublayer[1] ( x , lambda x : self.src_attn ( x, m ,m , src_mask ) )
        # tgt_mask should obey the time rule, yet src_mask needn`t
        return self.sublayer[2] ( x, self.feed_forward )


def subsequent_mask (size):
    """
    with input like 4 (an int), output will be tensor
        T F F F
        T T F F
        T T T F
        T T T T
    """
    attn_shape = (1 ,size ,size)
    subsequent_mask = (
            torch.triu( torch.ones( attn_shape ), diagonal = 1 ) 
            .type (torch.uint8 )
            )
    # triangular upper triu  diagonal = 1 means upshift 1 from diagonal
    return subsequent_mask == 0


def attention( query , key , value , mask = None , dropout = None ):
    d_query = query.size(-1)
    # Here, the variant query is of the shape (nbatch , seq_len_query , d_query)
    # And the variant key is of the shape (nbatch , seq_len_key , d_key)
    # That is , the vector q of (d_query) and the vector k of (d_key)
    # qTk suggests the attention weight(relativity) of q on k
    scores = torch.matmul(query , key . transpose (-2,-1)) / math.sqrt( d_query ) #Thus QKT
    if mask is not None:
        scores = scores.masked_fill ( mask == 0 , -1e9 )
        # (i,j) suggests i th query`s attention on j th key
        # for any j > i, i th query is not allowed to put attention on j th key
        # Thus only distills relativity prior to this word
    p_attn = scores.softmax( dim = -1 ) 
    if dropout is not None:
        p_attn = dropout( p_attn )
    return torch.matmul( p_attn , value ) , p_attn # The initial input has been updated to 
# merge the whole seq`s info into every word of it


class MultiHeadedAttention( nn.Module ):
    def __init__( self , h , d_model , dropout = 0.1 ): # h is number of heads
        super().__init__()
        assert d_model % h == 0
        self . d_query = d_model // h #divide size on the last dim
        self . h = h
        self.linears = clones ( nn.Linear ( d_model , d_model ) , 4 ) # To proj the last concatnated vec into the same size
        self.attn  = None
        self.dropout = nn.Dropout( p = dropout )

    def forward( self , query , key , value , mask = None ): # query of shape
        # (nbatch , seq_len_query , d_model)
        if mask is not None:
            mask = mask.unsqueeze(1)
            # from shape (nbatch , seq_len , seq_len ) to 
            # (nbatch , num_heads , seq_len , seq_len)
        nbatches = query.size(0) # batch size

        query, key, value = [
                lin(x) . view ( nbatches , -1 , self.h , self . d_query)
                # break the last dim of d_model into h * d_query with row first principle
                .transpose(1,2) # into shape ( nbatches , num_heads , seq_len , d_query )
                for lin, x in zip ( self.linears , (query , key , value) )
                ]
        # the lin here is the linear to proj x into h different heads

        x , self_attn = attention(
                query , key , value , mask = mask , dropout = self.dropout
                ) # the second output is the relativity (attention weight) tensor

        x = (
                x.transpose ( 1, 2 ) # num_heads and seq_len
                .contiguous()
                .view( nbatches , -1 , self.h * self.d_query ) # back to d_model
                )

        del query
        del key
        del value
        return self.linears[-1] (x) # the last linear used here


class PositionwiseFeedForward(nn.Module):

    def __init__ (self , d_model , d_ff , dropout = 0.1):
        super().__init__()
        self . w_1 = nn.Linear( d_model , d_ff )
        self . w_2 = nn.Linear( d_ff , d_model )
        self . dropout  = nn.Dropout(dropout)

    def forward( self , x ):
        return self.w_2 ( self.dropout ( self.w_1 (x) . relu() ) )


class Embeddings(nn.Module):
    def __init__( self , d_model , vocab ): # vocab here means size of vocabulary
        super().__init__()
        self.lut = nn.Embedding( vocab , d_model ) # look up table lut
        # input a list of index , output every row indexed with input
        self.d_model = d_model
        # and the lut is of the exact size as (vocab , d_model)

    def forward( self , x ):
        return self.lut(x) * math.sqrt (self . d_model)
    # from integer ( as index ) to vector (of size d_model)


class PositionalEncoding ( nn.Module ):

    def __init__( self , d_model , dropout , max_len = 5000 ): # max_len here means the max seq_len
        super().__init__()
        self.dropout = nn.Dropout( p = dropout )

        pe = torch.zeros (max_len , d_model) # PositionalEncoding PE
        position = torch.arange( 0 , max_len ) . unsqueeze (1)
        # that is (0 , 1, 2 , ... , max_len -1) into shape (max_len , 1)
        div_term  = torch.exp(
                torch.arange (0 , d_model , 2) # that is (0, 2, 4, ... , d_model)
                * - (math.log(10000.0) / d_model)
                )

        pe [ : , 0::2 ] = torch . sin( position * div_term )
        # broadcasted position will be like
        # 0 0 0 ... 0
        # 1 1 1 ... 1
        # ...
        # max_len-1 max_len-1 ... max_len-1
        # so said pos
        pe [ : , 1::2 ] = torch. cos (position * div_term) # rightside has one more colume 
        # than leftside , will be disgarded automatically

        pe = pe.unsqueeze(0) # now the shape of (1, max_len , d_model)
        self.register_buffer ("pe" , pe)# carry into GPU mem but not updated by grad

    def forward(self ,x):
        x = x + self.pe [: , :x.size(1)] . requires_grad_(False) 
        # x of shape (nbatches , seq_len , d_model)
        return self.dropout(x)
        

def make_model(
        src_vocab , tgt_vocab , N = 6 , d_model = 512 , d_ff = 2048 , h = 8 , dropout = 0.1
        ):
    c = copy.deepcopy
    attn = MultiHeadedAttention( h , d_model ) # with dropout not passed in
    ff = PositionwiseFeedForward ( d_model , d_ff, dropout )
    position = PositionalEncoding ( d_model , dropout ) # with max_len not passed in
    embed_share = Embeddings( d_model , tgt_vocab )
    model = EncoderDecoder( #(self , encoder , decoder , src_embed , tgt_embed , generator)
            Encoder( #(self , layer , N)
                EncoderLayer( #( self , size , self_attn , feed_forward , dropout )
                    d_model , c(attn) , c(ff) , dropout
                    ),
                N
            ),
            Decoder(
                DecoderLayer( #( self , size , self_attn , src_attn, feed_forward , dropout )
                    d_model , c(attn) , c(attn) , c(ff) , dropout
                    ),
                N
                ),
            nn.Sequential(
                embed_share,
                 # ( self , d_model , vocab)
                 # from (nbatches , seq_len) to ( nbatches , seq_len , d_model ) by lut
                c (position) # from and to (nbatches , seq_len , d_model) by plusing PE
                ),
            nn.Sequential(
                embed_share , # the size of vocab is used to build lut
                # as the size of (vocab , d_model)
                c (position)
                ),
            Generator (embed_share.lut.weight) # the vocab used here is to proj last dim
            # from d_model to tgt_vocab size
        )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if src_vocab == tgt_vocab:
        pass
        

    return model

