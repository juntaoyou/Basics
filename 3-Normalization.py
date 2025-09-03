import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.tensor, training=True):
        # x的形状: (batch_size, num_features, ...) 对于卷积层
        # 或 (batch_size, num_features) 对于全连接层
        
        if training:
            dims = (0,) + tuple(range(2, x.dim()))   
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)
            
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean = self.running_mean()
            var = self.running_var()
            
        x_normalized = ((x - mean[None,:,None,None]) if x.dim() == 4 else (x - mean[None, :])) /  \
        (torch.sqrt(var[None, :, None, None] + self.eps) if x.dim() == 4 else torch.sqrt(var[None, :] + self.eps))
        
        x_normalized = self.gamma[None,:,None,None] * x_normalized + self.beta[None,:,None,None] if x.dim() == 4 else self.gamma[None,:] * x_normalized + self.beta[None,:]
        
        return x_normalized
    
if __name__ == "__main__":
    # 测试BatchNormalization
    print("测试BatchNormalization:")
    batch_norm_custom = BatchNorm(num_features=3)
    batch_norm_torch = nn.BatchNorm2d(num_features=3)
    
    # 使用相同的输入
    x = torch.randn(2, 3, 4, 4)  # 随机输入 (batch_size=2, channels=3, height=4, width=4)
    
    # 切换到训练模式
    batch_norm_custom.train()
    batch_norm_torch.train()
    
    # 前向传播
    out_custom = batch_norm_custom(x)
    out_torch = batch_norm_torch(x)
    
    print(f"自定义实现输出形状: {out_custom.shape}")
    print(f"PyTorch内置实现输出形状: {out_torch.shape}")
    
    # # 测试LayerNormalization
    # print("\n测试LayerNormalization:")
    # layer_norm_custom = LayerNormalization(normalized_shape=5)
    # layer_norm_torch = nn.LayerNorm(normalized_shape=5)
    
    # # 使用相同的输入
    # x = torch.randn(2, 3, 5)  # 随机输入 (batch_size=2, seq_len=3, features=5)
    
    # # 前向传播
    # out_custom = layer_norm_custom(x)
    # out_torch = layer_norm_torch(x)
    
    # print(f"自定义实现输出形状: {out_custom.shape}")
    # print(f"PyTorch内置实现输出形状: {out_torch.shape}")
# https://www.doubao.com/thread/w3a24bd2b7b7cbf54


