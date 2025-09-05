import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, n_heads, seq_len_q, d_k]
            k: [batch_size, n_heads, seq_len_k, d_k]
            v: [batch_size, n_heads, seq_len_v, d_v] (seq_len_k == seq_len_v)
            mask: [batch_size, 1, seq_len_q, seq_len_k] 或 None
            
        Returns:
            output: [batch_size, n_heads, seq_len_q, d_v]
            attn: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = q.size(-1)
        # 计算注意力分数: (batch_size, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        output = torch.matmul(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, seq_len_q, d_model]
            k: [batch_size, seq_len_k, d_model]
            v: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k] 或 None
            
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attn: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        
        # 线性变换并分拆成多头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # 调整掩码形状以适应多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            
        # 计算注意力
        output, attn = self.attention(q, k, v, mask)
        
        # 拼接多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出线性变换
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output, attn

class PositionWiseFeedForward(nn.Module):
    """位置-wise前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数，比原论文的ReLU效果更好
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads,  dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.tensor, src_mask=None):
        attn_outputs, attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_outputs))
        
        ffn_outputs =  self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ffn_outputs))
        
        return x, attn
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads,  dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output, self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        cross_attn_output, cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        ffn_outputs = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ffn_outputs))
        
        return x, self_attn, cross_attn
if __name__ == "__main__":
    # 超参数设置
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    # 创建编码器层和解码器层实例
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
    
    # 生成随机输入
    batch_size = 32
    src_seq_len = 10
    tgt_seq_len = 15
    
    src = torch.randn(batch_size, src_seq_len, d_model)  # 编码器输入
    tgt = torch.randn(batch_size, tgt_seq_len, d_model)  # 解码器输入
    
    # 测试编码器层
    enc_output, enc_attn = encoder_layer(src)
    print(f"Encoder output shape: {enc_output.shape}")  # 应输出 torch.Size([32, 10, 512])
    print(f"Encoder attention shape: {enc_attn.shape}")  # 应输出 torch.Size([32, 8, 10, 10])
    
    # 测试解码器层
    dec_output, dec_self_attn, dec_cross_attn = decoder_layer(tgt, enc_output)
    print(f"Decoder output shape: {dec_output.shape}")  # 应输出 torch.Size([32, 15, 512])
    print(f"Decoder self-attention shape: {dec_self_attn.shape}")  # 应输出 torch.Size([32, 8, 15, 15])
    print(f"Decoder cross-attention shape: {dec_cross_attn.shape}")  # 应输出 torch.Size([32, 8, 15, 10])

# https://www.doubao.com/thread/w1ecfb532fb3519fd