import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from dataset import create_dataloaders

# 检查transformers是否可用
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  需要安装transformers库: pip install transformers")

class TransformersBasedTranslator(nn.Module):
    """基于Transformers分词器的翻译模型"""
    
    def __init__(self, tokenizer_name='bert-base-multilingual-cased', 
                 max_len=128, dropout=0.1, hidden_dim=512):
        super(TransformersBasedTranslator, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装transformers库")
        
        # 加载预训练的tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoder = AutoModel.from_pretrained(tokenizer_name)
        
        # 获取词汇表大小和隐藏维度
        self.vocab_size = len(self.tokenizer)
        self.d_model = self.encoder.config.hidden_size
        self.max_len = max_len
        
        # 解码器 - 使用LSTM + 注意力
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.decoder_lstm = nn.LSTM(
            self.d_model, hidden_dim, num_layers=2, 
            dropout=dropout, batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化新参数
        self.init_decoder_weights()
        
        print(f"模型初始化完成:")
        print(f"  - 编码器: {tokenizer_name}")
        print(f"  - 词汇表大小: {self.vocab_size}")
        print(f"  - 编码器维度: {self.d_model}")
        print(f"  - 解码器维度: {hidden_dim}")
    
    def init_decoder_weights(self):
        """初始化解码器权重"""
        for name, param in self.decoder_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def encode_text(self, text, max_length=None):
        """编码文本"""
        if max_length is None:
            max_length = self.max_len
        
        # 使用tokenizer编码
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids):
        """前向传播"""
        # 编码源序列
        encoder_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # 解码目标序列
        tgt_embeddings = self.decoder_embedding(tgt_input_ids)
        tgt_embeddings = self.dropout(tgt_embeddings)
        
        # LSTM解码
        lstm_output, _ = self.decoder_lstm(tgt_embeddings)
        
        # 注意力机制
        attended_output, _ = self.attention(
            query=lstm_output,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=~src_attention_mask.bool()
        )
        
        # 输出投影
        output = self.output_projection(attended_output)
        
        return output
    
    def generate(self, src_text, max_len=100):
        """生成翻译"""
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 编码源文本
            if isinstance(src_text, str):
                src_text = [src_text]
            
            encoded = self.encode_text(src_text, max_length=self.max_len)
            src_input_ids = encoded['input_ids'].to(device)
            src_attention_mask = encoded['attention_mask'].to(device)
            
            # 编码
            encoder_outputs = self.encoder(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # 初始化解码
            batch_size = src_input_ids.size(0)
            generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            generated[:, 0] = self.tokenizer.cls_token_id  # 使用CLS作为开始token
            
            hidden = None
            
            for _ in range(max_len):
                # 嵌入当前token
                current_emb = self.decoder_embedding(generated[:, -1:])
                
                # LSTM解码
                lstm_out, hidden = self.decoder_lstm(current_emb, hidden)
                
                # 注意力
                attended_out, _ = self.attention(
                    query=lstm_out,
                    key=encoder_hidden_states,
                    value=encoder_hidden_states,
                    key_padding_mask=~src_attention_mask.bool()
                )
                
                # 预测下一个token
                logits = self.output_projection(attended_out)
                next_token = torch.argmax(logits, dim=-1)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查结束条件
                if next_token.item() == self.tokenizer.sep_token_id:
                    break
            
            return generated

class TransformersTrainer:
    """基于Transformers的训练器"""
    
    def __init__(self, model, train_loader, valid_loader, 
                 lr=2e-5, device='cuda', save_dir='checkpoints_transformers'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器 - 对预训练参数使用不同学习率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.encoder.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': lr * 0.1  # 预训练部分使用更小学习率
            },
            {
                'params': [p for n, p in model.encoder.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': lr * 0.1
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if 'encoder' not in n],
                'weight_decay': 0.01,
                'lr': lr  # 新参数使用正常学习率
            }
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='logs_transformers')
    
    def convert_batch_to_transformers_format(self, batch):
        """将批次数据转换为transformers格式"""
        # 这里需要根据实际的batch格式进行调整
        english_ids = batch['english']
        chinese_ids = batch['chinese']
        
        # 创建attention mask
        src_attention_mask = (english_ids != self.model.tokenizer.pad_token_id).long()
        
        return {
            'src_input_ids': english_ids,
            'src_attention_mask': src_attention_mask,
            'tgt_input_ids': chinese_ids
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 转换数据格式
            formatted_batch = self.convert_batch_to_transformers_format(batch)
            
            src_input_ids = formatted_batch['src_input_ids'].to(self.device)
            src_attention_mask = formatted_batch['src_attention_mask'].to(self.device)
            tgt_input_ids = formatted_batch['tgt_input_ids'].to(self.device)
            
            # 准备解码器输入和目标
            decoder_input = tgt_input_ids[:, :-1]
            target = tgt_input_ids[:, 1:]
            
            # 前向传播
            output = self.model(src_input_ids, src_attention_mask, decoder_input)
            
            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
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
                formatted_batch = self.convert_batch_to_transformers_format(batch)
                
                src_input_ids = formatted_batch['src_input_ids'].to(self.device)
                src_attention_mask = formatted_batch['src_attention_mask'].to(self.device)
                tgt_input_ids = formatted_batch['tgt_input_ids'].to(self.device)
                
                decoder_input = tgt_input_ids[:, :-1]
                target = tgt_input_ids[:, 1:]
                
                output = self.model(src_input_ids, src_attention_mask, decoder_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Valid', avg_loss, epoch)
        return avg_loss
    
    def train(self, num_epochs=5):
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
                self.save_model(epoch, 'best_transformers_model.pth')
                print(f'Saved best model with valid loss: {valid_loss:.4f}')
        
        self.writer.close()
    
    def save_model(self, epoch, filename):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer_name': self.model.tokenizer.name_or_path,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

def main():
    """主训练函数 - Transformers版本"""
    if not TRANSFORMERS_AVAILABLE:
        print("❌ 需要安装transformers库: pip install transformers")
        return
    
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
    
    # 数据加载 - 使用transformers分词器
    print("Loading data with Transformers tokenizer...")
    try:
        train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
            train_path=train_path,
            valid_path=valid_path,
            batch_size=16,  # 较小的batch size
            max_len=64,
            use_transformers=True,
            tokenizer_name='bert-base-multilingual-cased'
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请确保数据文件存在且transformers库已正确安装")
        return
    
    # 创建模型
    print("Creating Transformers-based model...")
    model = TransformersBasedTranslator(
        tokenizer_name='bert-base-multilingual-cased',
        max_len=64,
        dropout=0.1,
        hidden_dim=512
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # 创建训练器
    trainer = TransformersTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=2e-5,
        device=device
    )
    
    # 开始训练
    print("Starting training with Transformers...")
    trainer.train(num_epochs=1)

if __name__ == "__main__":
    main()
