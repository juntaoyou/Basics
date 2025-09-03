import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    """
    普通位置编码（基于正弦余弦函数的实现）
    参考自 "Attention Is All You Need" 论文
    """
    def __init__(self, d_model:int, max_len: int = 5000):
        super(PositionalEmbedding,self).__init__()
        self.position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 0, 1::2] = torch.cos(self.position * self.div_term)
        self.register_buffer('positionalembedding', self.pe)
        
    def forward(self, x: torch.tensor):
        """
        Args:
            x: 输入张量，形状为 (seq_len, batch_size, d_model)
            
        Returns:
            加入位置编码后的张量，形状不变
        """
        x = x + self.pe[:x.size(0)]  # 加上位置编码
        return x
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        inv_freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base) / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._precompute_rope()
        
    def _precompute_rope(self):
        position = torch.arange(self.max_len, device = self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", position, self.inv_freq)
        self.register_buffer("cos_emb", freqs.cos())
        self.register_buffer("sin_emb", freqs.sin())
        
    def forward(self, x: torch.tensor):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            应用旋转位置编码后的张量，形状不变
        """
        
        batch_size, seq_len, d_model = x.shape
          
        x1 = x[..., ::2] # (batch_size, seq_len, d_model//2)
        x2 = x[..., 1::2] # (batch_size, seq_len, d_model//2)
        cos = self.cos_emb[:seq_len].unsqueeze(0) # (1, seq_len, d_model)
        sin = self.sin_emb[:seq_len].unsqueeze(0) # (1, seq_len, d_model)

        
        
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],dim = -1)
        
        return x_rot
        
if __name__ == "__main__":
    # 设置参数
    d_model = 512
    seq_len = 10
    batch_size = 2
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试普通位置编码
    pos_emb = PositionalEmbedding(d_model, max_len=100)
    # 注意：普通位置编码期望输入形状为 (seq_len, batch_size, d_model)
    x_pos = pos_emb(x.permute(1, 0, 2)).permute(1, 0, 2)
    print(f"普通位置编码输入形状: {x.shape}")
    print(f"普通位置编码输出形状: {x_pos.shape}")
    
    # 测试RoPE
    rope = RotaryPositionalEmbedding(d_model, max_len=100)
    x_rope = rope(x)
    print(f"RoPE输入形状: {x.shape}")
    print(f"RoPE输出形状: {x_rope.shape}")
    
    # 验证输出形状是否正确
    assert x_pos.shape == x.shape, "普通位置编码输出形状不正确"
    assert x_rope.shape == x.shape, "RoPE输出形状不正确"
    print("所有测试通过!")
    
    
# https://www.doubao.com/thread/w22265f79e955ae68