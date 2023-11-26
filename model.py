import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model : int, vocab_size : int,):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
class PostionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int,dropout : float):
        super().__init__()
        self.d_model = d_model #D
        self.seq_len = seq_len #L
        self.dropout = dropout

        self.dropout = nn.Dropout(dropout)

        #mat - shape (L,D)
        pe = torch.zeros((seq_len,d_model))
        #vec of len seq_len - basically index array for word/token (L,1)
        pos = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000)/d_model)) #diff from paper for num stability
        #apply sin/cos for even/odd
        pe[:,0::2] = torch.sin(pos*denom)
        pe[:,1::2] = torch.cos(pos*denom)
        #allow for batches make dim (1,L,D)
        pe = pe.unsqueeze(0)
        #store pe as it is not a learned param -> equivalent of storing in cache variable in 231n
        self.register_buffer('pe', pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
        
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model:int,d_ff:int,dropout:int):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h
        assert d_model%h==0,"d_model not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask==0,-1e9)
        
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores@value),attention_scores




    def forward(self,q,k,v,mask):
        
        #dim -> (batch,seq_len,d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #(batch,seq_len,d_model) -> (batch,seq_len,h,d_k) -> (batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self,features: int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self,features: int,self_attention_block: MultiHeadAttentionBlock,feed_forward_block:FeedForward,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self,features: int, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,features: int, self_attention_block: MultiHeadAttentionBlock,cross_attention_block: MultiHeadAttentionBlock,feed_forward_block:FeedForward,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask,):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self,features: int, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder:Encoder,decoder:Decoder, src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PostionalEncoding,tgt_pos:PostionalEncoding,projection_layer:ProjectionLayer) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)
    

def bulid_transformer(src_vocab_size:int, tgt_vocab_size: int, src_seq_len:int, tgt_seq_len:int,d_model:int = 512,N: int = 6,h:int = 8,dropout:float = 0.1,d_ff:int=2048) -> Transformer:
    #create Embedding layers

    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    #Create postitional encoding layers

    src_pos = PostionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PostionalEncoding(d_model,tgt_seq_len,dropout)

    #encoder blks
    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(d_model,encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)


    #encoder blks
    decoder_blocks = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(d_model,decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model,nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return transformer

