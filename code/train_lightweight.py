import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from dataset import create_dataloaders

class SimplePretrainedTranslator(nn.Module):
    """简化的预训练翻译模型 - 不依赖transformers库"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, 
                 hidden_dim=512, num_layers=2, dropout=0.1):
        super(SimplePretrainedTranslator, self).__init__()
        
        # 使用预训练的词嵌入（这里简化为可训练的嵌入）
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        
        # 编码器 - 使用双向LSTM
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # 解码器 - 使用单向LSTM + 注意力
        self.decoder = nn.LSTM(
            embedding_dim + hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # 注意力机制
        self.attention = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim * 2)
        self.attention_combine = nn.Linear(hidden_dim + hidden_dim * 2, embedding_dim)
        
        # 输出层
        self.output = nn.Linear(hidden_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 参数初始化
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, src, tgt):
        """前向传播"""
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # 编码
        src_emb = self.dropout(self.src_embedding(src))
        encoder_outputs, (hidden, cell) = self.encoder(src_emb)
        
        # 初始化解码器状态
        # 将双向LSTM的隐藏状态转换为单向LSTM的初始状态
        hidden = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
        cell = cell.view(self.encoder.num_layers, 2, batch_size, -1)
        
        # 取最后一层的前向和后向状态
        decoder_hidden = hidden[-1].transpose(0, 1).contiguous().view(batch_size, -1)
        decoder_cell = cell[-1].transpose(0, 1).contiguous().view(batch_size, -1)
        
        # 调整维度以匹配解码器
        decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        decoder_cell = decoder_cell.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        # 解码
        outputs = []
        decoder_input = self.dropout(self.tgt_embedding(tgt))
        
        for t in range(tgt_len):
            # 当前时间步的输入
            current_input = decoder_input[:, t:t+1, :]
            
            # 注意力权重
            attention_weights = torch.softmax(
                self.attention(torch.cat([
                    decoder_hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1),
                    encoder_outputs
                ], dim=2)), dim=1
            )
            
            # 注意力上下文
            context = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)
            
            # 结合输入和上下文
            combined_input = torch.cat([current_input, context], dim=2)
            combined_input = self.dropout(self.attention_combine(combined_input))
            
            # LSTM解码
            output, (decoder_hidden, decoder_cell) = self.decoder(
                combined_input, (decoder_hidden, decoder_cell)
            )
            
            # 输出投影
            output = self.output(output)
            outputs.append(output)
        
        return torch.cat(outputs, dim=1)
    
    def generate(self, src, max_len=100, sos_idx=2, eos_idx=3):
        """生成翻译"""
        self.eval()
        
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # 编码
            src_emb = self.src_embedding(src)
            encoder_outputs, (hidden, cell) = self.encoder(src_emb)
            
            # 初始化解码器状态
            hidden = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
            cell = cell.view(self.encoder.num_layers, 2, batch_size, -1)
            
            decoder_hidden = hidden[-1].transpose(0, 1).contiguous().view(batch_size, -1)
            decoder_cell = cell[-1].transpose(0, 1).contiguous().view(batch_size, -1)
            
            decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
            decoder_cell = decoder_cell.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
            
            # 初始化输出序列
            outputs = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            outputs[:, 0] = sos_idx
            
            for t in range(max_len):
                # 当前输入
                current_input = self.tgt_embedding(outputs[:, -1:])
                
                # 注意力
                attention_weights = torch.softmax(
                    self.attention(torch.cat([
                        decoder_hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1),
                        encoder_outputs
                    ], dim=2)), dim=1
                )
                
                context = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)
                
                # 结合输入和上下文
                combined_input = torch.cat([current_input, context], dim=2)
                combined_input = self.attention_combine(combined_input)
                
                # 解码
                output, (decoder_hidden, decoder_cell) = self.decoder(
                    combined_input, (decoder_hidden, decoder_cell)
                )
                
                # 预测下一个词
                next_token_logits = self.output(output)
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # 添加到输出序列
                outputs = torch.cat([outputs, next_token], dim=1)
                
                # 检查结束条件
                if torch.all(next_token == eos_idx):
                    break
            
            return outputs

class LightweightTrainer:
    """轻量级训练器"""
    
    def __init__(self, model, train_loader, valid_loader, vocab_en, vocab_zh, 
                 lr=1e-3, device='cuda', save_dir='checkpoints_lightweight'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.vocab_en = vocab_en
        self.vocab_zh = vocab_zh
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='logs_lightweight')
        
        # 反向词汇表
        self.id_to_en = {v: k for k, v in vocab_en.items()}
        self.id_to_zh = {v: k for k, v in vocab_zh.items()}
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            src = batch['english'].to(self.device)
            tgt = batch['chinese'].to(self.device)
            
            # 解码器输入和目标
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]
            
            # 前向传播
            output = self.model(src, decoder_input)
            
            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Train', avg_loss, epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.valid_loader)
        
        with torch.no_grad():
            for batch in self.valid_loader:
                src = batch['english'].to(self.device)
                tgt = batch['chinese'].to(self.device)
                
                decoder_input = tgt[:, :-1]
                target = tgt[:, 1:]
                
                output = self.model(src, decoder_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Valid', avg_loss, epoch)
        
        return avg_loss
    
    def generate_sample(self, src_text, max_len=50):
        """生成翻译样例"""
        self.model.eval()
        
        # 分词和转换
        tokens = src_text.lower().split()
        src_ids = [self.vocab_en.get(token, self.vocab_en['<unk>']) for token in tokens]
        src_tensor = torch.tensor([src_ids]).to(self.device)
        
        # 生成翻译
        with torch.no_grad():
            output = self.model.generate(src_tensor, max_len=max_len, 
                                       sos_idx=self.vocab_zh['<sos>'],
                                       eos_idx=self.vocab_zh['<eos>'])
        
        # 解码
        output_ids = output[0].cpu().numpy()
        tokens = [self.id_to_zh.get(id, '<unk>') for id in output_ids]
        
        # 移除特殊符号
        tokens = [token for token in tokens if token not in ['<sos>', '<eos>', '<pad>']]
        
        return ''.join(tokens)
    
    def train(self, num_epochs=8):
        """训练模型"""
        best_valid_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\n=== Epoch {epoch+1}/{num_epochs} ===')
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            valid_loss = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model(epoch, 'best_lightweight_model.pth')
                print(f'Saved best model with valid loss: {valid_loss:.4f}')
            
            # 生成样例翻译
            if (epoch + 1) % 2 == 0:
                sample_text = "This is a test sentence."
                translation = self.generate_sample(sample_text)
                print(f'Sample: {sample_text} -> {translation}')
        
        self.writer.close()
    
    def save_model(self, epoch, filename):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_en': self.vocab_en,
            'vocab_zh': self.vocab_zh,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

def main():
    """主训练函数 - 轻量级版本"""
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据加载 - 使用更小的配置进行快速实验
    print("Loading data...")
    train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
        train_path="data/translation2019zh_train.json",
        valid_path="data/translation2019zh_valid.json",
        batch_size=16,  # 适中的批大小
        max_len=32      # 更短的序列长度，加快训练
    )
    
    # 轻量级模型配置
    model_config = {
        'src_vocab_size': len(vocab_en),
        'tgt_vocab_size': len(vocab_zh),
        'embedding_dim': 256,    # 较小的嵌入维度
        'hidden_dim': 512,       # 较小的隐藏维度
        'num_layers': 2,         # 较少的层数
        'dropout': 0.1
    }
    
    # 创建模型
    print("Creating lightweight model...")
    model = SimplePretrainedTranslator(**model_config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:,} parameters')
    print(f'English vocab size: {len(vocab_en)}')
    print(f'Chinese vocab size: {len(vocab_zh)}')
    
    # 创建训练器
    trainer = LightweightTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocab_en=vocab_en,
        vocab_zh=vocab_zh,
        lr=1e-3,  # 较大的学习率，适合从头训练
        device=device
    )
    
    # 开始训练
    print("Starting lightweight training...")
    trainer.train(num_epochs=8)  # 适中的训练轮数

if __name__ == "__main__":
    main()
