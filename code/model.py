import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class TransformerTranslator(nn.Module):
    """Transformer翻译模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, max_len=512, dropout=0.1):
        super(TransformerTranslator, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, pad_idx=0):
        """创建填充掩码"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        """创建前瞻掩码"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src, src_mask=None):
        """编码器"""
        # 词嵌入 + 位置编码
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        src_emb = self.dropout(src_emb)
        
        # 编码器层
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """解码器"""
        # 词嵌入 + 位置编码
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.dropout(tgt_emb)
        
        # 解码器层
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def forward(self, src, tgt):
        """前向传播"""
        # 创建掩码
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        
        # 创建前瞻掩码
        seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(seq_len).to(tgt.device)
        tgt_mask = tgt_mask & look_ahead_mask
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate(self, src, max_len=100, sos_idx=2, eos_idx=3, pad_idx=0):
        """生成翻译"""
        self.eval()
        
        with torch.no_grad():
            # 编码源序列
            src_mask = self.create_padding_mask(src)
            encoder_output = self.encode(src, src_mask)
            
            # 初始化目标序列
            batch_size = src.size(0)
            tgt = torch.zeros(batch_size, 1, dtype=torch.long, device=src.device)
            tgt[:, 0] = sos_idx
            
            for i in range(max_len):
                # 创建掩码
                tgt_mask = self.create_padding_mask(tgt)
                look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
                tgt_mask = tgt_mask & look_ahead_mask
                
                # 解码
                decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
                
                # 预测下一个词
                next_token_logits = self.output_projection(decoder_output[:, -1, :])
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                
                # 添加到目标序列
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 检查是否生成结束符
                if torch.all(next_token == eos_idx):
                    break
            
            return tgt
