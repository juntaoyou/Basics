## 2025-09-02
## 内容：手撕MultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    实现Scaled Dot-Product Attention
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, mask=None):
        """
        参数:
            query: (batch_size, n_heads, seq_len_q, d_k)
            key: (batch_size, n_heads, seq_len_k, d_k)
            value: (batch_size, n_heads, seq_len_v, d_v)，通常seq_len_k = seq_len_v
            mask: (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, n_heads, seq_len_q, seq_len_k)
        返回:
            output: 注意力加权后的输出
            attn_weights: 注意力权重
        """
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)
            
        attn_weights = F.softmax(scores, dim = -1)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
class MultiHeadAttention(nn.Module):
    """
    实现Multi-Head Attention
    公式: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
          其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, d_model, n_heads):
        """
        参数:
            d_model: 模型的维度
            n_heads: 头的数量
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
    
    def forward(self, Q, K, V, mask=None):
        """
        参数:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)，通常seq_len_k = seq_len_v
            mask: (batch_size, seq_len_q, seq_len_k)
        返回:
            output: 多头注意力的输出 (batch_size, seq_len_q, d_model)
            attn_weights: 注意力权重 (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        
        batch_size = Q.shape[0]
        # 线性变换并分头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        Q = self.w_q(Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        output, attn_weights = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output, attn_weights
    
if __name__ == "__main__":
    # 超参数
    batch_size = 2
    seq_len = 5
    d_model = 64
    n_heads = 4
    
    x = torch.rand(batch_size, seq_len, d_model)
    
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask = torch.tril(mask)
    
    print("Testing ScaledDotProductAttention...")
    attn = ScaledDotProductAttention()
    
    # 模拟已经分头的Q, K, V
    q = torch.randn(batch_size, n_heads, seq_len, d_model//n_heads)
    k = torch.randn(batch_size, n_heads, seq_len, d_model//n_heads)
    v = torch.randn(batch_size, n_heads, seq_len, d_model//n_heads)
    
    output, attn_weights = attn(q, k, v)
    print(f"Scaled Dot-Product Output shape: {output.shape}")  # 应为 (batch_size, n_heads, seq_len, d_model//n_heads)
    print(f"Attention Weights shape: {attn_weights.shape}")    # 应为 (batch_size, n_heads, seq_len, seq_len)
    
    # 测试MultiHeadAttention
    print("\nTesting MultiHeadAttention...")
    multi_head_attn = MultiHeadAttention(d_model, n_heads)
    output, attn_weights = multi_head_attn(x, x, x, mask)  # 自注意力机制，Q=K=V
    print(f"Multi-Head Output shape: {output.shape}")      # 应为 (batch_size, seq_len, d_model)
    print(f"Multi-Head Attention Weights shape: {attn_weights.shape}")  # 应为 (batch_size, n_heads, seq_len, seq_len)

# Attention相关问题
# https://www.doubao.com/thread/wc659f2fbfaeae0ae