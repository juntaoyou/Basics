import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力实现"""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q: 查询张量 [batch_size, n_heads, seq_len_q, d_k]
            k: 键张量 [batch_size, n_heads, seq_len_k, d_k]
            v: 值张量 [batch_size, n_heads, seq_len_v, d_v] (seq_len_k == seq_len_v)
            mask: 掩码张量 [batch_size, 1, seq_len_q, seq_len_k] 或 None
            
        Returns:
            output: 注意力输出 [batch_size, n_heads, seq_len_q, d_v]
            attn: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = q.size(-1)
        # 计算注意力分数并缩放
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
    """多头注意力实现"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
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
        
    def forward(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q: 查询张量 [batch_size, seq_len_q, d_model]
            k: 键张量 [batch_size, seq_len_k, d_model]
            v: 值张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量 [batch_size, seq_len_q, seq_len_k] 或 None
            
        Returns:
            output: 多头注意力输出 [batch_size, seq_len_q, d_model]
            attn: 注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k]
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
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数，比原论文的ReLU效果更好
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            output: 前馈网络输出 [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 自注意力层
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: Tensor, 
        src_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: 输入张量 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码 [batch_size, src_seq_len, src_seq_len] 或 None
            
        Returns:
            output: 编码器层输出 [batch_size, src_seq_len, d_model]
            attn: 注意力权重 [batch_size, n_heads, src_seq_len, src_seq_len]
        """
        # 自注意力子层，带残差连接和层归一化
        attn_output, attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈神经网络子层，带残差连接和层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attn

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 掩码自注意力层（用于目标序列）
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 编码器-解码器注意力层
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: Tensor, 
        enc_output: Tensor, 
        src_mask: Optional[Tensor] = None, 
        tgt_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            enc_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码 [batch_size, 1, src_seq_len] 或 None
            tgt_mask: 目标序列掩码 [batch_size, tgt_seq_len, tgt_seq_len] 或 None
            
        Returns:
            output: 解码器层输出 [batch_size, tgt_seq_len, d_model]
            self_attn: 自注意力权重 [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
            cross_attn: 交叉注意力权重 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        """
        # 掩码自注意力子层
        self_attn_output, self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # 编码器-解码器注意力子层
        cross_attn_output, cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 前馈神经网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x, self_attn, cross_attn

class PositionalEncoding(nn.Module):
    """位置编码实现"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不参与梯度更新的参数
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 输入嵌入 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
            
        Returns:
            x: 加入位置编码后的嵌入 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        # 适配两种输入维度格式
        if x.size(0) != self.pe.size(0) and x.size(1) == self.pe.size(0):
            x = x + self.pe.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        else:
            x = x + self.pe[:x.size(0)]  # [seq_len, batch_size, d_model]
            
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: EncoderLayer,
        num_layers: int, 
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm
    
    def forward(
        self,
        src: Tensor, 
        src_mask: Optional[Tensor]=None
    ):
        
        output = src
        attn_weights = []
        for layer in self.layers:
            output, attn = layer(output,src_mask)
            attn_weights.append(attn)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output, attn_weights
    
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: DecoderLayer,
        num_layers: int, 
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm
    
    def forward(
        self,
        tgt: Tensor, 
        enc_output: Tensor, 
        tgt_mask: Optional[Tensor]=None,
        src_mask: Optional[Tensor]=None
    ):
        
        output = tgt
        self_attn_weights = []
        cross_attn_weights = []
        for layer in self.layers:
            output, self_attn, cross_attn = layer(output, enc_output, src_mask, tgt_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output, self_attn_weights, cross_attn_weights
    

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):   
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(d_model))
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, nn.LayerNorm(d_model))
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
      self,
      src: Tensor,
      tgt: Tensor,
      src_mask: Optional[Tensor]=None,  
      tgt_mask: Optional[Tensor]=None,
    )-> Tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.src_pos_encoding(src_emb)
        
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)   
        
        enc_output, enc_attn_weights = self.encoder(src_emb, src_mask) 
        dec_output, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        
        output = self.fc(dec_output)
        
        return output, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights

def generate_padding_mask(seq: Tensor, pad_idx: int) -> Tensor:
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)     

def generate_look_ahead_mask(seq_len: int, device: torch.device) -> Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    
    # 超参数设置
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1
    
    # 创建Transformer模型实例
    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        dropout
    )
    
    # 生成随机输入
    batch_size = 32
    src_seq_len = 10
    tgt_seq_len = 15
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 生成掩码
    pad_idx = 0
    src_mask = generate_padding_mask(src, pad_idx)
    tgt_mask = generate_padding_mask(tgt, pad_idx) & generate_look_ahead_mask(tgt_seq_len, tgt.device)
    
    # 前向传播测试
    output, enc_attn, dec_self_attn, dec_cross_attn = transformer(src, tgt, src_mask, tgt_mask)
    
    # 打印输出形状
    print(f"Transformer output shape: {output.shape}")  # 应输出 torch.Size([32, 15, 1000])
    print(f"Encoder attention layers: {len(enc_attn)}, first layer shape: {enc_attn[0].shape}")
    print(f"Decoder self-attention layers: {len(dec_self_attn)}, first layer shape: {dec_self_attn[0].shape}")
    print(f"Decoder cross-attention layers: {len(dec_cross_attn)}, first layer shape: {dec_cross_attn[0].shape}")
# https://www.doubao.com/thread/w1ecfb532fb3519fd