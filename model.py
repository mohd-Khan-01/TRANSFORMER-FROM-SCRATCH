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
    