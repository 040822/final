import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import re
from collections import Counter
import pickle
import os

class TranslationDataset(Dataset):
    """英译中翻译数据集"""
    
    def __init__(self, data_path, vocab_en=None, vocab_zh=None, max_len=128):
        """
        初始化数据集
        Args:
            data_path: JSON数据文件路径
            vocab_en: 英文词汇表
            vocab_zh: 中文词汇表
            max_len: 最大序列长度
        """
        self.data_path = data_path
        self.max_len = max_len
        self.data = []
        
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
    
    def load_data(self):
        """加载JSON数据"""
        print(f"Loading data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # 清理数据
                    english = self.clean_text(item['english'])
                    chinese = self.clean_text(item['chinese'])
                    
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
        """英文分词"""
        return text.lower().split()
    
    def tokenize_chinese(self, text):
        """中文分词"""
        return list(jieba.cut(text))
    
    def build_vocab(self):
        """构建词汇表"""
        print("Building vocabulary...")
        
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
        """预处理数据，转换为数字序列"""
        print("Preprocessing data...")
        processed_data = []
        
        for item in self.data:
            # 分词
            en_tokens = self.tokenize_english(item['english'])
            zh_tokens = self.tokenize_chinese(item['chinese'])
            
            # 转换为数字序列
            en_ids = self.tokens_to_ids(en_tokens, self.vocab_en)
            zh_ids = self.tokens_to_ids(zh_tokens, self.vocab_zh)
            
            # 添加特殊符号
            zh_ids = [self.vocab_zh['<sos>']] + zh_ids + [self.vocab_zh['<eos>']]
            
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
        return [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def save_vocab(self, vocab_path):
        """保存词汇表"""
        with open(vocab_path, 'wb') as f:
            pickle.dump((self.vocab_en, self.vocab_zh), f)
    
    def load_vocab(self, vocab_path):
        """加载词汇表"""
        with open(vocab_path, 'rb') as f:
            self.vocab_en, self.vocab_zh = pickle.load(f)

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

def create_dataloaders(train_path, valid_path, batch_size=32, max_len=128):
    """创建数据加载器"""
    
    # 加载训练集
    train_dataset = TranslationDataset(train_path, max_len=max_len)
    
    # 保存词汇表
    train_dataset.save_vocab('vocab.pkl')
    
    # 加载验证集（使用相同的词汇表）
    valid_dataset = TranslationDataset(
        valid_path, 
        vocab_en=train_dataset.vocab_en,
        vocab_zh=train_dataset.vocab_zh,
        max_len=max_len
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
    # 测试数据加载
    train_path = "data/translation2019zh_train.json"
    valid_path = "data/translation2019zh_valid.json"
    
    train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
        train_path, valid_path, batch_size=4, max_len=64
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"English vocab size: {len(vocab_en)}")
    print(f"Chinese vocab size: {len(vocab_zh)}")
    
    # 查看第一个batch
    for batch in train_loader:
        print("Sample batch:")
        print(f"English shape: {batch['english'].shape}")
        print(f"Chinese shape: {batch['chinese'].shape}")
        break