import torch
import torch.nn as nn
from einops import rearrange

# We use einsum everywhere for the clarity purpose

checkpoint = torch.load('./1v1_v0_pretrained.pt')
w_q_weigts = checkpoint['w_q_weigts']  # (L, H, N0, G)
w_k_weigts = checkpoint['w_k_weigts']

L, H, N, G = w_q_weigts.shape

eps = 1e-4
batch_size = 3
num_tokens = 5

# just create a random input
x_0 = torch.randn([batch_size, num_tokens, N])

# Compute attention using weights:
q = torch.einsum('lhng,btn -> lbhtg', w_q_weigts, x_0)
k = torch.einsum('lhng,btn -> lbhtg', w_k_weigts, x_0)
# attn = torch.matmul(q, k.transpose(-1, -2))
attn = torch.einsum('lbhtg,lbhsg -> lbhts', q, k)

# Create W = KQ^T
W = torch.einsum('lhig,lhjg -> lhij', w_k_weigts, w_q_weigts)
attn2 = torch.einsum('lhij,btj -> lbhti', W, x_0)
attn2 = torch.einsum('lbhti,bsi -> lbhts', attn2, x_0)

print('Test 1')
assert (attn - attn2 < eps).all()
print('done')

# OR:
W2 = torch.einsum('lhig,lhjg -> lhij', w_q_weigts, w_k_weigts)
attn3 = torch.einsum('lhij,bti -> lbhtj', W2, x_0)
attn3 = torch.einsum('lbhti,bsi -> lbhts', attn3, x_0)

print('Test 2')
assert (attn - attn3 < eps).all()
print('done')

# Compare to actual attn computation in the model
to_q = nn.Linear(N, G * H, bias=False)
to_k = nn.Linear(N, G * H, bias=False)

to_q.weight.data = rearrange(w_q_weigts[0], 'h n0 g -> (h g) n0', h=H)
to_k.weight.data = rearrange(w_k_weigts[0], 'h n0 g -> (h g) n0', h=H)

q = rearrange(to_q(x_0), 'b t (h g) -> b h t g', h=H)
k = rearrange(to_k(x_0), 'b t (h g) -> b h t g', h=H)

attn_code = torch.matmul(q, k.transpose(-1, -2))

print('Test 3')
assert (attn - attn_code < eps).all()
print('done')

print('All test passed')