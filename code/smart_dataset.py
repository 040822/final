import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy as np
from sklearn.model_selection import train_test_split

# 尝试导入相关库
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers库可用")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers库不可用")

try:
    import spacy
    SPACY_AVAILABLE = True
    print("✓ SpaCy库可用")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  SpaCy库不可用")

class CachedTranslationDataset(Dataset):
    """支持缓存的英译中翻译数据集"""
    
    def __init__(self, data_path, vocab_en=None, vocab_zh=None, max_len=128, 
                 tokenizer_type='transformers', tokenizer_name='bert-base-multilingual-cased',
                 cache_dir='cache', sample_ratio=1.0, min_freq=2):
        """
        初始化数据集
        Args:
            data_path: JSON数据文件路径
            vocab_en: 英文词汇表
            vocab_zh: 中文词汇表
            max_len: 最大序列长度
            tokenizer_type: 分词器类型 ('transformers', 'spacy', 'basic')
            tokenizer_name: 预训练分词器名称
            cache_dir: 缓存目录
            sample_ratio: 数据采样比例 (0.0-1.0)
            min_freq: 最小词频阈值
        """
        self.data_path = data_path
        self.max_len = max_len
        self.tokenizer_type = tokenizer_type
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.sample_ratio = sample_ratio
        self.min_freq = min_freq
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存标识
        self.cache_key = self._generate_cache_key()
        
        # 初始化分词器
        self.tokenizer = self._init_tokenizer()
        
        # 检查缓存或加载数据
        if self._load_from_cache():
            print("✓ 从缓存加载数据成功")
        else:
            print("缓存不存在，开始处理数据...")
            self._process_data()
            self._save_to_cache()
        
        # 设置词汇表
        if vocab_en is None or vocab_zh is None:
            if not hasattr(self, 'vocab_en'):
                self.vocab_en, self.vocab_zh = self._build_vocab_if_needed()
        else:
            self.vocab_en = vocab_en
            self.vocab_zh = vocab_zh
    
    def _generate_cache_key(self):
        """生成缓存键"""
        key_str = f"{os.path.basename(self.data_path)}_{self.max_len}_{self.tokenizer_type}_{self.tokenizer_name}_{self.sample_ratio}_{self.min_freq}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _init_tokenizer(self):
        """初始化分词器"""
        if self.tokenizer_type == 'transformers' and TRANSFORMERS_AVAILABLE:
            try:
                print(f"加载Transformers分词器: {self.tokenizer_name}")
                return AutoTokenizer.from_pretrained(self.tokenizer_name)
            except Exception as e:
                print(f"Transformers分词器加载失败: {e}")
                return None
        elif self.tokenizer_type == 'spacy' and SPACY_AVAILABLE:
            try:
                print("加载SpaCy分词器")
                # 尝试加载多语言模型
                try:
                    return spacy.load("xx_core_web_sm")  # 多语言小模型
                except OSError:
                    return spacy.load("en_core_web_sm")  # 英文模型
            except Exception as e:
                print(f"SpaCy分词器加载失败: {e}")
                return None
        else:
            print("使用基础分词器")
            return None
    
    def _load_from_cache(self):
        """从缓存加载数据"""
        cache_file = os.path.join(self.cache_dir, f"dataset_{self.cache_key}.pkl")
        vocab_file = os.path.join(self.cache_dir, f"vocab_{self.cache_key}.pkl")
        
        if os.path.exists(cache_file) and os.path.exists(vocab_file):
            try:
                print("正在从缓存加载数据...")
                with open(cache_file, 'rb') as f:
                    self.data = pickle.load(f)
                with open(vocab_file, 'rb') as f:
                    vocab_data = pickle.load(f)
                    self.vocab_en = vocab_data['vocab_en']
                    self.vocab_zh = vocab_data['vocab_zh']
                print(f"缓存加载完成，共 {len(self.data)} 个样本")
                return True
            except Exception as e:
                print(f"缓存加载失败: {e}")
                return False
        return False
    
    def _save_to_cache(self):
        """保存数据到缓存"""
        cache_file = os.path.join(self.cache_dir, f"dataset_{self.cache_key}.pkl")
        vocab_file = os.path.join(self.cache_dir, f"vocab_{self.cache_key}.pkl")
        
        try:
            print("正在保存数据到缓存...")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            with open(vocab_file, 'wb') as f:
                pickle.dump({
                    'vocab_en': self.vocab_en,
                    'vocab_zh': self.vocab_zh
                }, f)
            print("缓存保存完成")
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def _process_data(self):
        """处理原始数据"""
        print(f"开始处理数据文件: {self.data_path}")
        
        # 分块读取大文件
        raw_data = []
        chunk_size = 10000
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            chunk = []
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    english = item.get('english', '').strip()
                    chinese = item.get('chinese', '').strip()
                    
                    if len(english) > 0 and len(chinese) > 0:
                        chunk.append({'english': english, 'chinese': chinese})
                    
                    if len(chunk) >= chunk_size:
                        raw_data.extend(chunk)
                        chunk = []
                        if i % 50000 == 0:
                            print(f"已处理 {i} 行数据...")
                
                except Exception as e:
                    continue
            
            # 处理最后一块
            if chunk:
                raw_data.extend(chunk)
        
        print(f"原始数据加载完成，共 {len(raw_data)} 个样本")
        
        # 数据采样
        if self.sample_ratio < 1.0:
            sample_size = int(len(raw_data) * self.sample_ratio)
            raw_data = np.random.choice(raw_data, sample_size, replace=False).tolist()
            print(f"数据采样完成，保留 {len(raw_data)} 个样本")
        
        # 构建词汇表（如果需要）
        if self.tokenizer_type in ['basic', 'spacy'] or not self.tokenizer:
            self.vocab_en, self.vocab_zh = self._build_vocab(raw_data)
        else:
            # 使用预训练分词器的词汇表
            self.vocab_en = dict(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else {}
            self.vocab_zh = self.vocab_en  # 多语言模型共享词汇表
        
        # 并行预处理数据
        self._preprocess_parallel(raw_data)
    
    def _build_vocab(self, raw_data):
        """构建词汇表"""
        print("构建词汇表...")
        from collections import Counter
        
        en_counter = Counter()
        zh_counter = Counter()
        
        for item in raw_data[:10000]:  # 只用前10000个样本构建词汇表以加速
            en_tokens = self._tokenize_text(item['english'], 'en')
            zh_tokens = self._tokenize_text(item['chinese'], 'zh')
            
            en_counter.update(en_tokens)
            zh_counter.update(zh_tokens)
        
        # 创建词汇表
        vocab_en = self._create_vocab_dict(en_counter)
        vocab_zh = self._create_vocab_dict(zh_counter)
        
        print(f"英文词汇表大小: {len(vocab_en)}")
        print(f"中文词汇表大小: {len(vocab_zh)}")
        
        return vocab_en, vocab_zh
    
    def _create_vocab_dict(self, counter):
        """创建词汇表字典"""
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        
        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                vocab[word] = len(vocab)
        
        return vocab
    
    def _tokenize_text(self, text, lang='en'):
        """分词文本"""
        if self.tokenizer_type == 'transformers' and self.tokenizer:
            return self.tokenizer.tokenize(text)
        elif self.tokenizer_type == 'spacy' and self.tokenizer:
            doc = self.tokenizer(text)
            return [token.text for token in doc]
        else:
            # 基础分词
            if lang == 'zh':
                try:
                    import jieba
                    return list(jieba.cut(text))
                except ImportError:
                    return list(text)  # 字符级分词
            else:
                return text.lower().split()
    
    def _preprocess_parallel(self, raw_data):
        """并行预处理数据"""
        print("开始并行预处理数据...")
        
        # 分块处理
        num_workers = min(multiprocessing.cpu_count(), 8)
        chunk_size = len(raw_data) // num_workers
        chunks = [raw_data[i:i+chunk_size] for i in range(0, len(raw_data), chunk_size)]
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
            
            processed_data = []
            for i, future in enumerate(futures):
                chunk_result = future.result()
                processed_data.extend(chunk_result)
                print(f"完成第 {i+1}/{len(futures)} 块处理")
        
        self.data = processed_data
        print(f"预处理完成，最终数据量: {len(self.data)}")
    
    def _process_chunk(self, chunk):
        """处理数据块"""
        processed_chunk = []
        
        for item in chunk:
            try:
                # 分词和编码
                if self.tokenizer_type == 'transformers' and self.tokenizer:
                    # 使用Transformers编码
                    en_encoding = self.tokenizer.encode_plus(
                        item['english'],
                        max_length=self.max_len,
                        padding=False,
                        truncation=True,
                        return_tensors=None
                    )
                    zh_encoding = self.tokenizer.encode_plus(
                        item['chinese'],
                        max_length=self.max_len,
                        padding=False,
                        truncation=True,
                        return_tensors=None
                    )
                    en_ids = en_encoding['input_ids']
                    zh_ids = zh_encoding['input_ids']
                else:
                    # 传统方法
                    en_tokens = self._tokenize_text(item['english'], 'en')
                    zh_tokens = self._tokenize_text(item['chinese'], 'zh')
                    
                    en_ids = [self.vocab_en.get(token, self.vocab_en['<unk>']) for token in en_tokens]
                    zh_ids = [self.vocab_zh.get(token, self.vocab_zh['<unk>']) for token in zh_tokens]
                    
                    # 添加特殊符号
                    zh_ids = [self.vocab_zh['<sos>']] + zh_ids + [self.vocab_zh['<eos>']]
                
                # 长度检查
                if len(en_ids) <= self.max_len and len(zh_ids) <= self.max_len:
                    processed_chunk.append({
                        'english': en_ids,
                        'chinese': zh_ids
                    })
            
            except Exception as e:
                continue
        
        return processed_chunk
    
    def _build_vocab_if_needed(self):
        """如果需要则构建词汇表"""
        if hasattr(self, 'vocab_en') and hasattr(self, 'vocab_zh'):
            return self.vocab_en, self.vocab_zh
        else:
            # 从已处理的数据构建词汇表（应该在缓存中）
            return self.vocab_en, self.vocab_zh
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def smart_collate_fn(batch):
    """智能批处理函数"""
    english_batch = [torch.tensor(item['english'], dtype=torch.long) for item in batch]
    chinese_batch = [torch.tensor(item['chinese'], dtype=torch.long) for item in batch]
    
    # 动态填充
    english_batch = pad_sequence(english_batch, batch_first=True, padding_value=0)
    chinese_batch = pad_sequence(chinese_batch, batch_first=True, padding_value=0)
    
    return {
        'input_ids': english_batch,
        'labels': chinese_batch,
        'attention_mask': (english_batch != 0).long()
    }

def create_smart_dataloaders(data_path, test_size=0.1, batch_size=32, max_len=128,
                           tokenizer_type='transformers', tokenizer_name='bert-base-multilingual-cased',
                           cache_dir='cache', sample_ratio=1.0, num_workers=4):
    """
    创建智能数据加载器
    Args:
        data_path: 数据文件路径
        test_size: 测试集比例
        batch_size: 批大小
        max_len: 最大序列长度
        tokenizer_type: 分词器类型
        tokenizer_name: 分词器名称
        cache_dir: 缓存目录
        sample_ratio: 数据采样比例
        num_workers: 数据加载工作进程数
    """
    print("🚀 创建智能数据加载器...")
    
    # 创建完整数据集
    full_dataset = CachedTranslationDataset(
        data_path=data_path,
        max_len=max_len,
        tokenizer_type=tokenizer_type,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
        sample_ratio=sample_ratio
    )
    
    # 分割数据集
    dataset_size = len(full_dataset)
    test_size_abs = int(dataset_size * test_size)
    train_size_abs = dataset_size - test_size_abs
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size_abs, test_size_abs],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ 数据集分割完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=smart_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=smart_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader, full_dataset.vocab_en, full_dataset.vocab_zh

if __name__ == "__main__":
    # 测试数据加载器
    print("=== 测试智能数据加载器 ===")
    
    train_loader, test_loader, vocab_en, vocab_zh = create_smart_dataloaders(
        data_path="data/translation2019zh_train.json",
        test_size=0.1,
        batch_size=4,
        max_len=64,
        tokenizer_type='transformers',
        tokenizer_name='bert-base-multilingual-cased',
        cache_dir='smart_cache',
        sample_ratio=0.01,  # 只用1%的数据测试
        num_workers=2
    )
    
    print(f"✓ 训练批次数: {len(train_loader)}")
    print(f"✓ 测试批次数: {len(test_loader)}")
    print(f"✓ 英文词汇表大小: {len(vocab_en)}")
    print(f"✓ 中文词汇表大小: {len(vocab_zh)}")
    
    # 测试一个批次
    for i, batch in enumerate(train_loader):
        print(f"✓ 批次 {i+1} 形状:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        if i >= 1:  # 只测试前2个批次
            break
    
    print("✓ 数据加载器测试完成！")
