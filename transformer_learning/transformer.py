import torch
from torch import nn
import torch.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask is not None:
            # mask = torch.tril(torch.ones(time, time, dtype=bool))
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        return output

class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,maxlen,device):
        super(PositionalEmbedding,self).__init__()
        self.encoding=torch.zeros(maxlen,d_model,device=device)
        self.encoding.requires_grad_(False)
        pos=torch.arange(0,maxlen,device=device)
        pos=pos.float().unsqueeze(1)
        
        
        
def main():
    re=torch.tensor([1,2,3,4,5,6])
    result=re.view(3,-1,1)
    print(result.shape)
    print(result.squeeze(0).shape)
    print(result.squeeze(1).shape)
    
    print(result.squeeze(2).shape)
    X=torch.randn(128,64,512)
    print(X.shape)

    d_model=512
    n_head=8
    attention=MultiHeadAttention(d_model,n_head)
    output=attention(X,X,X)
    print(output)
    
if __name__ == '__main__':
    main()