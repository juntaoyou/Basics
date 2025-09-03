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
        middlevar = self.position * self.div_term
        # print(middlevar.shape)
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
    # print(pos_emb.position.shape)
    # print(pos_emb.div_term.shape)
    # print(pos_emb.pe.shape)
    print(f"普通位置编码输入形状: {x.shape}")
    print(f"普通位置编码输出形状: {x_pos.shape}")
    
    # # 测试RoPE
    # rope = RoPE(d_model, max_len=100)
    # x_rope = rope(x)
    # print(f"RoPE输入形状: {x.shape}")
    # print(f"RoPE输出形状: {x_rope.shape}")
    
    # # 验证输出形状是否正确
    # assert x_pos.shape == x.shape, "普通位置编码输出形状不正确"
    # assert x_rope.shape == x.shape, "RoPE输出形状不正确"
    # print("所有测试通过!")
# https://www.doubao.com/thread/w22265f79e955ae68