import torch
import torch.nn as nn
import math

class Input_Embeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(d_model,vocab_size)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
        
class Positional_Encoding(nn.Module):
    def __init__(self,d_model:int,seq_length:int,dropout:int):
        super().__init__()
        self.d_model=d_model
        self.seq_length=seq_length
        self.dropout=nn.Dropout(dropout)
        #lets create a matrix of shape (d_model,seq_length)
        pe=torch.zeros(d_model,seq_length)
        #create a vector of shape (seq_len,1)
        position=torch.arange(0,seq_length,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(1000.0)/d_model))
        #applying sin to the even position and cos to the odd position
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1:2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    def forward(self,x):
        x=x+(self.pe[:,:[x].shape[1],:]).requires_grad(False)
        return self.dropout(x)
    
class Layer_Normilization(nn.Module):
    def __init___(self,eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))#multiplied 
        self.bias=nn.Parameter(torch.ones(1))#added
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
    
class Feedforward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,drop_out:float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)#w1 and  b2
        self.dropout=nn.Dropout(drop_out)
        self.linear_2=nn.Linear(d_ff,d_model)#w1 and b2
    
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):#h is the number of the heads
        super().__init__()
        self.d_model=d_model
        self.h=h
        self.drop_out=nn.Dropout()
        assert d_model % h ==0 ,"dimension of the model is not divisable by the number of the heads"
        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
    @staticmethod
    def attention(query,key,value,mask,drop_out:nn.Dropout):
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=attention_scores.soft_max(dim=-1)
        if drop_out is not None:
            attention_scores=drop_out(attention_scores)
        return (attention_scores@value),attention_scores
    
    def forward(self,q,k,v,mask):
        query=self.w_q(q) 
        key=self.w_k(k)
        value=self.w_v(v)       
        query=query.view(query[0],query[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key[0],key[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value[0],value[1],self.h,self.d_k).transpose(1,2)
        
        x,self.attention_scores=MultiHeadAttention.attention(query,key,value,mask,self.drop_out)
        x=x.transpose(1,2).contigous().view(x.shape[0],-1,self.h*self.d_k)
        
        return self.w_o(x)
class Residual_connection(nn.module):
    def __init__(self,dropout:nn.Dropout):
        super().__init__()
        self.dropout=dropout
        self.norm=Layer_Normilization()
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: Feedforward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([Residual_connection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = Layer_Normilization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,cross_attention_block:MultiHeadAttention,dropout:nn.Dropout,feed_forward_block:Feedforward):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection=nn.Module(Residual_connection(dropout) for _ in  range (3))
        
    def forward(self,x,src_mask,trg_mask):
        x=self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connection[1](x,lambda x: self.cross_attention_block(x,x,x,trg_mask))
        x=self.residual_connection[2](x,self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()
        self.layers=layers
        self.normalization=Layer_Normilization()
    
    def forward(self,x,encoder_output,src_mask,target_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,target_mask)
        return self.norm(x)

class projectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)