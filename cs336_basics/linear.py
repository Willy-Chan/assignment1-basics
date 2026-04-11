import torch
import torch.nn as nn
import einops
from einops import reduce, rearrange

# einops
# einsum() is just __batched__ matrix multiplication
#       Y = einsum(D, A,"... d_in, d_out d_in -> ... d_out")

# rearrange just lets you shuffle dimensions
# dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
# images_rearr = rearrange(images,  "b height width channel -> b 1 height width channel")
# dimmed_images = images_rearr * dim_value

# math is y = Wx, but we implement as y = xW^T

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        # no bias as per instructions
        self.device = device
        self.dtype = dtype

        # initialization
        mean = 0
        var = float(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean, var, -3 * var, 3 * var)


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("...i, io -> ...o", x, self.weight)

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype
        mean = 0.0
        var = 1.0
        nn.init.trunc_normal_(self.weight, mean, var, -3, 3)

    
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]



class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    
    def forward(
        self,
        x: torch.Tensor # (batch, sequence_length, d_model)
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)     # upcast to FP32 to avoid overflow

        # RMSNorm computation
        rms_term = einops.reduce(x_fp32 ** 2, 'b s d -> b s 1', 'mean')
        rms_term = torch.sqrt(rms_term + self.eps)

        return (x_fp32 / rms_term * self.weight).to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden_layer: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_hidden_layer, device, dtype)
        self.w2 = Linear(d_hidden_layer, d_model, device, dtype)
        self.w3 = Linear(d_model, d_hidden_layer, device, dtype)

    
    def forward(
        self,
        x: torch.Tensor # (d_model)
    ) -> torch.Tensor:
        w1x = self.w1(x)   # d_hidden_layer
        return self.w2((w1x * torch.sigmoid(w1x)) * self.w3(x))


# TODO:
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.d_k = d_k
        
        exponent = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)  # [0, 2, 4, ... d_k - 2] are my exponents
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)       # sequence positions
        freqs = torch.outer(t, 1.0 / (theta ** (exponent / d_k)))

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
  
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)

        x_pairs = einops.rearrange(x, '... (d j) -> ... d j', j=2)
        x1 = x_pairs[..., 0:1]
        x2 = x_pairs[..., 1:2]

        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return einops.rearrange(x_rotated, '... d j -> ... (d j)')



def stable_softmax_on_vector(
    tensor: torch.Tensor, 
    dim: int
) -> torch.Tensor:
    max_val = torch.max(tensor, dim=dim, keepdim=True)
    exponentials = torch.exp(tensor - max_val.values)
    sum_exp = torch.sum(exponentials, dim=dim, keepdim=True)
    return exponentials / sum_exp


import math
# EINSUM NOTE: you just specify 2 dimensions: as long as one of the dimensions in each part is shared, the result will be fine (assumed to be summed over)
# It does auto-implicit-transpose for you!

# elem-wise multiply: '...id, ...id -> ...id'
# elem-wise multiply then reducesum: '...id, ...id -> ...i'
# matmul/dot product of every i with every j: '...id, ...jd -> ...ij'
def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask=None
) -> torch.Tensor:
    d_k = q.shape[-1]
    # d_v = v.shape(-1)

    # scores = qk^T
    # queries are (iseq_len, dim) and keys are (jseq_len, dim)
    scores = torch.einsum('...id, ...jd -> ...ij', q, k) / math.sqrt(d_k)

    # mask out: -inf ensures that softmax is 0 for those entries
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    max_scores = torch.max(scores, dim=-1, keepdim=True).values
    exponentials = torch.exp(scores - max_scores)
    attn_probs = exponentials / torch.sum(exponentials, dim=-1, keepdim=True)

    output = torch.einsum('...ij, ...jd -> ...id', attn_probs, v)
    
    return output


class CausalMultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    # REARRANGE EINOP:
    #   (b, s, d) -> (b, s, (h, d)) implicitly via the rearrange einop!!!
    #   IF you do this, you need to fix the math by setting h=self.num_heads!!!
    # IMPORTANT NOTE OF REARRANGE EINOP + PARENTHESES: 
    #   can either use parentheses in the input (b, s, (h, d)) -> splitting up dims
    #   Or as output something -> (b, s, (h, d))   - combining dims
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        rope_layer : RotaryPositionalEmbedding = None,
        token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        seq_len = x.shape[-2]

        # split the QKV into heads
        q = rearrange(q, '... s (h d) -> ... h s d', h=self.num_heads)      # (batch ... seq_len, d_model) is what gets fed into the attention block
        k = rearrange(k, '... s (h d) -> ... h s d', h=self.num_heads)
        v = rearrange(v, '... s (h d) -> ... h s d', h=self.num_heads)

        # apply rope to q and k
        if rope_layer is not None:
            q = rope_layer(q, token_positions)
            k = rope_layer(k, token_positions)
        
        # default mask is causal mask
        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        
        output_values = scaled_dot_product_attention(q, k, v, mask=mask)

        output_values = rearrange(output_values, '... h s d -> ... s (h d)', h=self.num_heads)

        return self.out_proj(output_values)


# 𝑦 = 𝑥 + MultiHeadSelfAttention(RMSNorm(𝑥))
class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.attention_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention_layer = CausalMultiheadAttention(d_model, num_heads, device=device, dtype=dtype)

        self.ffw_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffw_layer = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor, 
        rope_layer : RotaryPositionalEmbedding = None,
        token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        res = x
        x = self.attention_norm(x)
        x = self.attention_layer(x, mask=mask, rope_layer=rope_layer, token_positions=token_positions)
        x = res + x

        res = x
        x = self.ffw_norm(x)
        x = self.ffw_layer(x)
        x = res + x

        return x