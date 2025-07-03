import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
from dataset import create_dataloaders

class PretrainedTranslationModel(nn.Module):
    """基于预训练模型的翻译系统"""
    
    def __init__(self, pretrained_model_name='bert-base-multilingual-cased', 
                 tgt_vocab_size=50000, max_len=128, dropout=0.1):
        super(PretrainedTranslationModel, self).__init__()
        
        # 加载预训练的多语言BERT作为编码器
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        # 获取编码器的隐藏维度
        self.d_model = self.encoder.config.hidden_size
        
        # 解码器部分 - 简化的LSTM解码器
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, self.d_model)
        self.decoder_lstm = nn.LSTM(self.d_model, self.d_model, num_layers=2, 
                                   dropout=dropout, batch_first=True)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(self.d_model, num_heads=8, 
                                             dropout=dropout, batch_first=True)
        
        # 输出层
        self.output_projection = nn.Linear(self.d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化解码器参数
        self.init_decoder_weights()
    
    def init_decoder_weights(self):
        """初始化解码器权重"""
        for name, param in self.decoder_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def encode(self, input_ids, attention_mask=None):
        """编码输入序列"""
        # 使用预训练模型编码
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    
    def decode_step(self, tgt_input, encoder_output, hidden_state=None, encoder_mask=None):
        """解码一步"""
        # 目标序列嵌入
        tgt_emb = self.decoder_embedding(tgt_input)
        tgt_emb = self.dropout(tgt_emb)
        
        # LSTM解码
        lstm_out, hidden_state = self.decoder_lstm(tgt_emb, hidden_state)
        
        # 注意力机制
        attended_output, _ = self.attention(
            query=lstm_out,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=encoder_mask
        )
        
        # 输出投影
        output = self.output_projection(attended_output)
        
        return output, hidden_state
    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids):
        """前向传播"""
        # 编码
        encoder_output = self.encode(src_input_ids, src_attention_mask)
        
        # 创建编码器掩码
        encoder_mask = ~src_attention_mask.bool() if src_attention_mask is not None else None
        
        # 解码
        output, _ = self.decode_step(tgt_input_ids, encoder_output, 
                                   encoder_mask=encoder_mask)
        
        return output
    
    def generate(self, src_input_ids, src_attention_mask, max_len=100, 
                sos_idx=2, eos_idx=3, pad_idx=0):
        """生成翻译"""
        self.eval()
        
        with torch.no_grad():
            batch_size = src_input_ids.size(0)
            device = src_input_ids.device
            
            # 编码源序列
            encoder_output = self.encode(src_input_ids, src_attention_mask)
            encoder_mask = ~src_attention_mask.bool() if src_attention_mask is not None else None
            
            # 初始化目标序列
            tgt_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            tgt_input[:, 0] = sos_idx
            
            hidden_state = None
            
            for i in range(max_len):
                # 解码一步
                output, hidden_state = self.decode_step(
                    tgt_input[:, -1:], encoder_output, hidden_state, encoder_mask
                )
                
                # 预测下一个词
                next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
                
                # 添加到目标序列
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                
                # 检查是否生成结束符
                if torch.all(next_token == eos_idx):
                    break
            
            return tgt_input

class PretrainedTranslationTrainer:
    """基于预训练模型的翻译训练器"""
    
    def __init__(self, model, train_loader, valid_loader, vocab_zh, 
                 lr=2e-5, device='cuda', save_dir='checkpoints_pretrained'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.vocab_zh = vocab_zh
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器 - 对预训练参数使用较小的学习率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.encoder.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': lr * 0.1  # 预训练部分使用更小的学习率
            },
            {
                'params': [p for n, p in model.encoder.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': lr * 0.1
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if 'encoder' not in n and not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': lr  # 新增参数使用正常学习率
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if 'encoder' not in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': lr
            }
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='logs_pretrained')
        
        # 反向词汇表
        self.id_to_zh = {v: k for k, v in vocab_zh.items()}
    
    def prepare_batch(self, batch):
        """准备批次数据"""
        # 英文使用预训练tokenizer
        english_texts = []
        for ids in batch['english']:
            # 从ids转回文本（这里需要原始文本，实际使用时可能需要调整）
            # 简化处理：直接使用tokenizer编码
            text = " ".join([str(id.item()) for id in ids if id.item() != 0])
            english_texts.append(text)
        
        # 使用预训练tokenizer编码英文
        encoded = self.model.tokenizer(
            english_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'src_input_ids': encoded['input_ids'].to(self.device),
            'src_attention_mask': encoded['attention_mask'].to(self.device),
            'tgt_input_ids': batch['chinese'].to(self.device)
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 准备数据
            prepared_batch = self.prepare_batch(batch)
            
            src_input_ids = prepared_batch['src_input_ids']
            src_attention_mask = prepared_batch['src_attention_mask']
            tgt_input_ids = prepared_batch['tgt_input_ids']
            
            # 解码器输入和目标
            decoder_input = tgt_input_ids[:, :-1]
            target = tgt_input_ids[:, 1:]
            
            # 前向传播
            output = self.model(src_input_ids, src_attention_mask, decoder_input)
            
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
            if batch_idx % 50 == 0:  # 更频繁的输出
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
                prepared_batch = self.prepare_batch(batch)
                
                src_input_ids = prepared_batch['src_input_ids']
                src_attention_mask = prepared_batch['src_attention_mask']
                tgt_input_ids = prepared_batch['tgt_input_ids']
                
                decoder_input = tgt_input_ids[:, :-1]
                target = tgt_input_ids[:, 1:]
                
                output = self.model(src_input_ids, src_attention_mask, decoder_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Valid', avg_loss, epoch)
        
        return avg_loss
    
    def train(self, num_epochs=5):  # 预训练模型通常需要更少的epoch
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
                self.save_model(epoch, 'best_pretrained_model.pth')
                print(f'Saved best model with valid loss: {valid_loss:.4f}')
            
            # 每个epoch都保存
            self.save_model(epoch, f'checkpoint_epoch_{epoch+1}.pth')
        
        self.writer.close()
    
    def save_model(self, epoch, filename):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_zh': self.vocab_zh,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

def main():
    """主训练函数 - 预训练模型版本"""
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据加载 - 使用更小的配置
    print("Loading data...")
    train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
        train_path="data/translation2019zh_train.json",
        valid_path="data/translation2019zh_valid.json",
        batch_size=8,   # 减小batch size
        max_len=64      # 减小序列长度
    )
    
    print(f"Chinese vocab size: {len(vocab_zh)}")
    
    # 创建基于预训练模型的翻译系统
    print("Creating pretrained model...")
    model = PretrainedTranslationModel(
        pretrained_model_name='bert-base-multilingual-cased',
        tgt_vocab_size=len(vocab_zh),
        max_len=64,
        dropout=0.1
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # 创建训练器
    trainer = PretrainedTranslationTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocab_zh=vocab_zh,
        lr=2e-5,  # 预训练模型通常使用较小的学习率
        device=device
    )
    
    # 开始训练
    print("Starting training with pretrained model...")
    trainer.train(num_epochs=5)  # 更少的训练轮数

if __name__ == "__main__":
    main()
