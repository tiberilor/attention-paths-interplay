import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class NoFFTransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., freeze_qk=False):
        super().__init__()
        # self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, freeze_qk=freeze_qk)
        # self.la2 = nn.LayerNorm(feats)

    def forward(self, x, x_0=None):
        if x_0 is not None:
            out = self.msa(x, x_0) + x
        else:
            out = self.msa(x) + x
        # out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., use_softmax=True, freeze_qk=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        self.use_softmax = use_softmax

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        if freeze_qk:
            # Freeze Q, K
            for param in self.q.parameters():
                param.requires_grad = False
            for param in self.k.parameters():
                param.requires_grad = False

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_0=None):
        b, n, f = x.size()
        if x_0 is not None:
            q = self.q(x_0).view(b, n, self.head, self.feats//self.head).transpose(1,2)
            k = self.k(x_0).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        else:
            q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
            k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        if self.use_softmax:
            score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        else:
            score = torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
