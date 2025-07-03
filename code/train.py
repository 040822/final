import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from dataset import create_dataloaders
from model import TransformerTranslator

class TranslationTrainer:
    """翻译模型训练器"""
    
    def __init__(self, model, train_loader, valid_loader, vocab_en, vocab_zh, 
                 lr=1e-4, device='cuda', save_dir='checkpoints'):
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
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='logs')
        
        # 反向词汇表（用于解码）
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
            
            # 输入和目标（解码器输入不包含最后一个token）
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]  # 目标不包含第一个token（SOS）
            
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
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
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
    
    def generate_sample(self, src_text, max_len=100):
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
    
    def train(self, num_epochs=20):
        """训练模型"""
        best_valid_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\n=== Epoch {epoch+1}/{num_epochs} ===')
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            valid_loss = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step(valid_loss)
            
            print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            
            # 保存最佳模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model(epoch, 'best_model.pth')
                print(f'Saved best model with valid loss: {valid_loss:.4f}')
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(epoch, f'checkpoint_epoch_{epoch+1}.pth')
            
            # 生成样例翻译
            if (epoch + 1) % 5 == 0:
                sample_text = "This is a test sentence for translation."
                translation = self.generate_sample(sample_text)
                print(f'Sample Translation: {sample_text} -> {translation}')
        
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
    
    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

def main():
    """主训练函数"""
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据加载
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)  # 确保当前工作目录是脚本所在
    print("Current working directory:", os.getcwd())
    print("root_dir:", root_dir)
    train_path = os.path.join(root_dir, "data/translation2019zh_train.json")
    valid_path = os.path.join(root_dir, "data/translation2019zh_valid.json")
    print(f"Train path: {train_path}")
    print(f"Valid path: {valid_path}")
    print("Loading data...")
    train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
        train_path=train_path,
        valid_path=valid_path,
        batch_size=16,
        max_len=128
    )
    
    # 模型参数
    model_config = {
        'src_vocab_size': len(vocab_en),
        'tgt_vocab_size': len(vocab_zh),
        'd_model': 512,
        'n_heads': 8,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'd_ff': 2048,
        'max_len': 512,
        'dropout': 0.1
    }
    
    # 创建模型
    print("Creating model...")
    model = TransformerTranslator(**model_config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:,} parameters')
    
    # 创建训练器
    trainer = TranslationTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocab_en=vocab_en,
        vocab_zh=vocab_zh,
        lr=1e-4,
        device=device
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train(num_epochs=20)

if __name__ == "__main__":
    main()
