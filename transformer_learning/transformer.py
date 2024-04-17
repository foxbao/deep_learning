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
        score = score.permute(0, 2, 1, 3).contiguous().view(
            batch, time, dimension)

        output = self.w_combine(score)
        return output


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(
            vocab_size, d_model, padding_idx=1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        self.encoding.requires_grad_(False)
        pos = torch.arange(0, maxlen, device=device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.drop1(x)
        x = self.norm1(x+_x)
        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.no2(x+_x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)
            x = self.dropout2(x)
            x = self.norm2(x+_x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+_x)


class Encoder(nn.Module):
    def __init__(self, env_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(
            env_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
             for _ in range(n_layer)]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(
            dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
             for _ in range(n_layer)]
        )
        self.fc = nn.Linear(d_model)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc(dec)
        return dec


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_len, d_model, n_heads, ffn_hidden, n_layers, drop_prob, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model,
                               ffn_hidden, n_heads, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model,
                               ffn_hidden, n_heads, n_layers, drop_prob, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device=device
        
    def make_pad_mask(self):
        pass
    
    def make_casual_mask(self):
        pass
    def forward(self,src,trg):
        pass


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
        
def main():
    X = torch.randn(128, 64, 512)
    src = torch.load('tensor_src.pt')
    src = torch.cat((src, torch.ones(src.shape[0], 2, dtype=torch.int)), dim=-1)
    trg = torch.load('tensor_trg.pt')
    # d_model = 512
    # n_head = 8
    # attention = MultiHeadAttention(d_model, n_head)
    # output = attention(X, X, X)
    # print(output)


if __name__ == '__main__':
    main()
