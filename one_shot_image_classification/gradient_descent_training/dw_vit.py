import torch
import torch.nn as nn
import torchsummary

from layers import TransformerEncoder

class DepthwiseViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8):
        super(DepthwiseViT, self).__init__()
        # hidden=384
        # hidden should be one third of original ViT.

        self.patch = patch # number of patches in one row(or col)
        self.patch_size = img_size//self.patch
        f = self.patch_size**2 # patch vec length

        self.emb = nn.Linear(f, hidden) # (b, c, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, 1, (self.patch**2)+1, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )


    def forward(self, x):
        out = self._to_words(x)
        out = torch.cat([self.cls_token.repeat(out.size(0),1,1,1), self.emb(out)],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out[:,0]
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, c, n, hw/n)
        """
        b,c,_,_ = x.size()
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).reshape(b,c,self.patch**2,-1)
        # out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out


if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    net = DepthwiseViT(in_c=c, num_classes= 10, img_size=h, patch=16, dropout=0.1, num_layers=7, hidden=384//3, head=12, mlp_hidden=384)
    # out = net(x)
    # out.mean().backward()
    torchsummary.summary(net, (c,h,w))
    # print(out.shape)
    