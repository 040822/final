"""
多模型架构定义
包含BiLSTM、Transformer、轻量级模型和预训练模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

try:
    from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Attention(nn.Module):
    """注意力机制"""
    
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [seq_len, batch_size, hidden_dim]
        
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        # 重复hidden状态
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力权重
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]
        
        return F.softmax(attention, dim=1)

class BiLSTMTranslator(nn.Module):
    """双向LSTM翻译模型"""
    
    def __init__(self, en_vocab_size, zh_vocab_size, embedding_dim=512, 
                 hidden_dim=512, num_layers=2, dropout=0.1, max_len=128):
        super(BiLSTMTranslator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        
        # 编码器
        self.en_embedding = nn.Embedding(en_vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # 解码器
        self.zh_embedding = nn.Embedding(zh_vocab_size, embedding_dim, padding_idx=0)
        self.decoder = nn.LSTM(
            embedding_dim + hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = Attention(hidden_dim)
        
        # 输出层
        self.out = nn.Linear(hidden_dim * 3, zh_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    
    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # 编码器
        src_emb = self.en_embedding(src)  # [batch_size, src_len, embedding_dim]
        encoder_outputs, (hidden, cell) = self.encoder(src_emb)
        
        # 合并双向LSTM的输出
        encoder_outputs = encoder_outputs[:, :, :self.hidden_dim] + encoder_outputs[:, :, self.hidden_dim:]
        
        # 初始化解码器状态
        decoder_hidden = hidden[-1].unsqueeze(0)  # 使用编码器最后一层的hidden
        decoder_cell = cell[-1].unsqueeze(0)
        
        # 解码器
        outputs = []
        input_token = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        for t in range(tgt_len):
            # 嵌入当前输入
            tgt_emb = self.zh_embedding(input_token)  # [batch_size, 1, embedding_dim]
            
            # 计算注意力
            attn_weights = self.attention(decoder_hidden.squeeze(0), encoder_outputs.transpose(0, 1))
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, hidden_dim*2]
            
            # 解码器输入：嵌入 + 上下文
            decoder_input = torch.cat([tgt_emb.squeeze(1), context], dim=1).unsqueeze(1)
            
            # 解码器前向传播
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            
            # 输出预测
            output = torch.cat([
                decoder_output.squeeze(1),
                context
            ], dim=1)
            
            output = self.out(self.dropout(output))
            outputs.append(output)
            
            # 下一个输入
            if t < tgt_len - 1:
                input_token = tgt[:, t + 1].unsqueeze(1)
        
        return torch.stack(outputs, dim=1)  # [batch_size, tgt_len, vocab_size]

class TransformerTranslator(nn.Module):
    """原始Transformer翻译模型"""
    
    def __init__(self, en_vocab_size, zh_vocab_size, d_model=512, nhead=8, 
                 num_layers=6, dropout=0.1, max_len=128):
        super(TransformerTranslator, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 嵌入层
        self.en_embedding = nn.Embedding(en_vocab_size, d_model, padding_idx=0)
        self.zh_embedding = nn.Embedding(zh_vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.out = nn.Linear(d_model, zh_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, pad_idx=0):
        """创建padding mask"""
        return (seq == pad_idx)
    
    def create_look_ahead_mask(self, size):
        """创建look-ahead mask"""
        mask = torch.triu(torch.ones(size, size)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        batch_size, tgt_len = tgt.size()
        device = src.device
        
        # 嵌入和位置编码
        src_emb = self.en_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.zh_embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # 创建mask
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        tgt_mask = self.create_look_ahead_mask(tgt_len).to(device)
        
        # Transformer前向传播
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        
        # 输出层
        output = self.out(self.dropout(output))
        
        return output

class LightweightTranslator(nn.Module):
    """轻量级翻译模型（基于预训练编码器）"""
    
    def __init__(self, tokenizer_name='bert-base-multilingual-cased', 
                 max_len=128, hidden_dim=512, dropout=0.1):
        super(LightweightTranslator, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装transformers库")
        
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        
        # 加载预训练编码器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoder = AutoModel.from_pretrained(tokenizer_name)
        
        # 冻结编码器参数（可选）
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        self.d_model = self.encoder.config.hidden_size
        vocab_size = len(self.tokenizer)
        
        # 解码器
        self.decoder_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len)
        
        # 简化的解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # 输出层
        self.out = nn.Linear(self.d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def create_masks(self, src, tgt):
        """创建注意力mask"""
        # 源序列padding mask
        src_padding_mask = (src == self.tokenizer.pad_token_id)
        
        # 目标序列padding mask
        tgt_padding_mask = (tgt == self.tokenizer.pad_token_id)
        
        # 目标序列causal mask
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device)) == 1
        tgt_mask = tgt_mask.transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        
        return src_padding_mask, tgt_padding_mask, tgt_mask
    
    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask=None):
        # 编码器
        encoder_outputs = self.encoder(
            input_ids=src_ids,
            attention_mask=src_mask
        )
        memory = encoder_outputs.last_hidden_state  # [batch_size, src_len, d_model]
        
        # 解码器嵌入和位置编码
        tgt_emb = self.decoder_embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # 创建mask
        src_padding_mask, tgt_padding_mask, causal_mask = self.create_masks(src_ids, tgt_ids)
        
        # 解码器
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # 输出层
        output = self.out(self.dropout(output))
        
        return output

class PretrainedTranslator(nn.Module):
    """基于预训练翻译模型的翻译器"""
    
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-zh', max_len=128, dropout=0.1):
        super(PretrainedTranslator, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装transformers库")
        
        self.max_len = max_len
        
        # 加载预训练翻译模型
        try:
            self.model = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        except:
            # 如果Marian模型不可用，使用通用模型
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 添加dropout层（可选）
        self.dropout = nn.Dropout(dropout)
        
        # 微调最后几层（可选）
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # 
        # # 只训练最后几层
        # for param in self.model.decoder.layers[-2:].parameters():
        #     param.requires_grad = True
    
    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask=None):
        """前向传播"""
        # 使用预训练模型的前向传播
        outputs = self.model(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=tgt_ids,
            decoder_attention_mask=tgt_mask,
            return_dict=True
        )
        
        # 应用dropout（如果需要）
        logits = self.dropout(outputs.logits)
        
        return logits
    
    def generate(self, src_ids, src_mask, max_length=None, num_beams=4):
        """生成翻译"""
        if max_length is None:
            max_length = self.max_len
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=src_ids,
                attention_mask=src_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        return outputs

# 模型工厂函数
def create_model(model_type, **kwargs):
    """模型工厂函数"""
    if model_type == 'bilstm':
        return BiLSTMTranslator(**kwargs)
    elif model_type == 'transformer':
        return TransformerTranslator(**kwargs)
    elif model_type == 'lightweight':
        return LightweightTranslator(**kwargs)
    elif model_type == 'pretrained':
        return PretrainedTranslator(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    src_len = 20
    tgt_len = 25
    vocab_size = 1000
    
    # 测试BiLSTM模型
    print("测试BiLSTM模型...")
    model = BiLSTMTranslator(
        en_vocab_size=vocab_size,
        zh_vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2
    )
    
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    
    output = model(src, tgt)
    print(f"BiLSTM输出形状: {output.shape}")
    
    # 测试Transformer模型
    print("\n测试Transformer模型...")
    model = TransformerTranslator(
        en_vocab_size=vocab_size,
        zh_vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_layers=2
    )
    
    output = model(src, tgt)
    print(f"Transformer输出形状: {output.shape}")
    
    print("\n模型测试完成!")
