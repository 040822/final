import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import re
from collections import Counter
import pickle
import os

# 尝试导入transformers，如果失败则使用传统方法
try:
    from transformers import AutoTokenizer, BertTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers库可用，将使用预训练分词器")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers库不可用，将使用传统分词方法")

class TranslationDataset(Dataset):
    """英译中翻译数据集 - 支持Transformers分词器"""
    
    def __init__(self, data_path, vocab_en=None, vocab_zh=None, max_len=128, 
                 use_transformers=True, tokenizer_name='bert-base-multilingual-cased'):
        """
        初始化数据集
        Args:
            data_path: JSON数据文件路径
            vocab_en: 英文词汇表
            vocab_zh: 中文词汇表
            max_len: 最大序列长度
            use_transformers: 是否使用transformers分词器
            tokenizer_name: 预训练分词器名称
        """
        self.data_path = data_path
        self.max_len = max_len
        self.data = []
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        
        # 初始化分词器
        self.init_tokenizers(tokenizer_name)
        
        # 加载数据
        self.load_data()
        
        # 构建或加载词汇表
        if vocab_en is None or vocab_zh is None:
            self.vocab_en, self.vocab_zh = self.build_vocab()
        else:
            self.vocab_en = vocab_en
            self.vocab_zh = vocab_zh
        
        # 预处理数据
        self.preprocess_data()
    
    def init_tokenizers(self, tokenizer_name):
        """初始化分词器"""
        if self.use_transformers:
            try:
                print(f"正在加载预训练分词器: {tokenizer_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print("✓ 预训练分词器加载成功")
            except Exception as e:
                print(f"⚠️  预训练分词器加载失败: {e}")
                print("回退到传统分词方法")
                self.use_transformers = False
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def load_data(self):
        """加载JSON数据"""
        print(f"Loading data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # 清理数据 数据已经被清理，为了加快速度不用再清理一遍
                    #english = self.clean_text(item['english'])
                    #chinese = self.clean_text(item['chinese'])
                    
                    english = item['english']
                    chinese = item['chinese']
                    
                    if len(english) > 0 and len(chinese) > 0:
                        self.data.append({
                            'english': english,
                            'chinese': chinese
                        })
                except Exception as e:
                    continue
        print(f"Loaded {len(self.data)} samples")
    
    def clean_text(self, text):
        """清理文本数据"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:\'"-]', '', text)
        return text.strip()
    
    def tokenize_english(self, text):
        """英文分词 - 支持transformers分词器"""
        if self.use_transformers and self.tokenizer:
            # 使用预训练分词器
            tokens = self.tokenizer.tokenize(text)
            return tokens
        else:
            # 传统分词方法
            return text.lower().split()
    
    def tokenize_chinese(self, text):
        """中文分词 - 支持transformers分词器"""
        if self.use_transformers and self.tokenizer:
            # 使用预训练分词器（对中文也有很好的支持）
            tokens = self.tokenizer.tokenize(text)
            return tokens
        else:
            # 传统jieba分词
            return list(jieba.cut(text))
    
    def encode_text_with_transformers(self, text, max_length=None):
        """使用transformers分词器编码文本"""
        if not self.use_transformers or not self.tokenizer:
            return None
        
        if max_length is None:
            max_length = self.max_len
        
        # 使用分词器编码
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        return encoded['input_ids']
    
    def build_vocab(self):
        """构建词汇表 - 支持transformers分词器"""
        print("Building vocabulary...")
        
        if self.use_transformers and self.tokenizer:
            print("使用预训练分词器构建词汇表...")
            # 使用预训练分词器的词汇表
            vocab_en = dict(self.tokenizer.vocab)
            vocab_zh = dict(self.tokenizer.vocab)  # 多语言模型共享词汇表
            
            print(f"Pretrained vocab size: {len(vocab_en)}")
            return vocab_en, vocab_zh
        else:
            print("使用传统方法构建词汇表...")
            # 统计词频
            en_counter = Counter()
            zh_counter = Counter()
            
            for item in self.data:
                en_tokens = self.tokenize_english(item['english'])
                zh_tokens = self.tokenize_chinese(item['chinese'])
                
                en_counter.update(en_tokens)
                zh_counter.update(zh_tokens)
            
            # 构建词汇表（保留高频词）
            vocab_en = self.create_vocab_dict(en_counter, min_freq=2)
            vocab_zh = self.create_vocab_dict(zh_counter, min_freq=2)
            
            print(f"English vocab size: {len(vocab_en)}")
            print(f"Chinese vocab size: {len(vocab_zh)}")
            
            return vocab_en, vocab_zh
    
    def create_vocab_dict(self, counter, min_freq=2):
        """创建词汇表字典"""
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,  # 开始符号
            '<eos>': 3   # 结束符号
        }
        
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)
        
        return vocab
    
    def preprocess_data(self):
        """预处理数据，转换为数字序列 - 支持transformers分词器"""
        print("Preprocessing data...")
        processed_data = []
        
        if self.use_transformers and self.tokenizer:
            print("使用预训练分词器预处理数据...")
            
            for item in self.data:
                try:
                    # 使用transformers分词器直接编码
                    en_ids = self.encode_text_with_transformers(item['english'])
                    zh_ids = self.encode_text_with_transformers(item['chinese'])
                    
                    if en_ids and zh_ids and len(en_ids) <= self.max_len and len(zh_ids) <= self.max_len:
                        processed_data.append({
                            'english': en_ids,
                            'chinese': zh_ids
                        })
                except Exception as e:
                    continue
        else:
            print("使用传统方法预处理数据...")
            
            for item in self.data:
                # 分词
                en_tokens = self.tokenize_english(item['english'])
                zh_tokens = self.tokenize_chinese(item['chinese'])
                
                # 转换为数字序列
                en_ids = self.tokens_to_ids(en_tokens, self.vocab_en)
                zh_ids = self.tokens_to_ids(zh_tokens, self.vocab_zh)
                
                # 添加特殊符号（仅在传统模式下）
                if not self.use_transformers:
                    zh_ids = [self.vocab_zh.get('<sos>', 2)] + zh_ids + [self.vocab_zh.get('<eos>', 3)]
                
                # 长度限制
                if len(en_ids) <= self.max_len and len(zh_ids) <= self.max_len:
                    processed_data.append({
                        'english': en_ids,
                        'chinese': zh_ids
                    })
        
        self.data = processed_data
        print(f"Preprocessed {len(self.data)} samples")
    
    def tokens_to_ids(self, tokens, vocab):
        """将tokens转换为ids"""
        if self.use_transformers and self.tokenizer:
            # 如果使用transformers，tokens已经是ids了
            return tokens
        else:
            # 传统方法：查找词汇表
            return [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens]
    
    def ids_to_tokens(self, ids):
        """将ids转换为tokens - 用于解码"""
        if self.use_transformers and self.tokenizer:
            return self.tokenizer.convert_ids_to_tokens(ids)
        else:
            # 传统方法需要反向词汇表
            id_to_token = {v: k for k, v in self.vocab_zh.items()}
            return [id_to_token.get(id, '<unk>') for id in ids]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def save_vocab(self, vocab_path):
        """保存词汇表和分词器配置"""
        save_data = {
            'vocab_en': self.vocab_en,
            'vocab_zh': self.vocab_zh,
            'use_transformers': self.use_transformers,
            'tokenizer_name': getattr(self.tokenizer, 'name_or_path', None) if self.tokenizer else None
        }
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"词汇表已保存到: {vocab_path}")
    
    def load_vocab(self, vocab_path):
        """加载词汇表和分词器配置"""
        with open(vocab_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.vocab_en = save_data['vocab_en']
        self.vocab_zh = save_data['vocab_zh']
        
        # 如果保存时使用了transformers，尝试重新加载
        if save_data.get('use_transformers', False) and save_data.get('tokenizer_name'):
            self.init_tokenizers(save_data['tokenizer_name'])

def collate_fn(batch):
    """批处理函数"""
    english_batch = [torch.tensor(item['english']) for item in batch]
    chinese_batch = [torch.tensor(item['chinese']) for item in batch]
    
    # 动态填充
    english_batch = pad_sequence(english_batch, batch_first=True, padding_value=0)
    chinese_batch = pad_sequence(chinese_batch, batch_first=True, padding_value=0)
    
    return {
        'english': english_batch,
        'chinese': chinese_batch
    }

def create_dataloaders(train_path, valid_path, batch_size=32, max_len=128, 
                      use_transformers=True, tokenizer_name='bert-base-multilingual-cased'):
    """创建数据加载器 - 支持transformers分词器"""
    
    print(f"创建数据加载器 - transformers: {use_transformers}")
    
    # 加载训练集
    train_dataset = TranslationDataset(
        train_path, 
        max_len=max_len,
        use_transformers=use_transformers,
        tokenizer_name=tokenizer_name
    )
    
    # 保存词汇表
    train_dataset.save_vocab('vocab.pkl')
    
    # 加载验证集（使用相同的词汇表和分词器配置）
    valid_dataset = TranslationDataset(
        valid_path, 
        vocab_en=train_dataset.vocab_en,
        vocab_zh=train_dataset.vocab_zh,
        max_len=max_len,
        use_transformers=train_dataset.use_transformers,
        tokenizer_name=tokenizer_name
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, train_dataset.vocab_en, train_dataset.vocab_zh

if __name__ == "__main__":
    # 测试数据加载 - 同时测试传统方法和transformers方法
    train_path = "data/translation2019zh_train.json"
    valid_path = "data/translation2019zh_valid.json"
    
    print("=== 测试1: 使用Transformers分词器 ===")
    try:
        train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
            train_path, valid_path, batch_size=4, max_len=64, 
            use_transformers=True, tokenizer_name='bert-base-multilingual-cased'
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Valid batches: {len(valid_loader)}")
        print(f"English vocab size: {len(vocab_en)}")
        print(f"Chinese vocab size: {len(vocab_zh)}")
        
        # 查看第一个batch
        for batch in train_loader:
            print("Sample batch (Transformers):")
            print(f"English shape: {batch['english'].shape}")
            print(f"Chinese shape: {batch['chinese'].shape}")
            break
    except Exception as e:
        print(f"Transformers方法失败: {e}")
        
        print("\n=== 测试2: 使用传统分词方法 ===")
        train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
            train_path, valid_path, batch_size=4, max_len=64, 
            use_transformers=False
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Valid batches: {len(valid_loader)}")
        print(f"English vocab size: {len(vocab_en)}")
        print(f"Chinese vocab size: {len(vocab_zh)}")
        
        # 查看第一个batch
        for batch in train_loader:
            print("Sample batch (Traditional):")
            print(f"English shape: {batch['english'].shape}")
            print(f"Chinese shape: {batch['chinese'].shape}")
            break