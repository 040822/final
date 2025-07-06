#!/usr/bin/env python3
"""
英文到中文翻译模型训练和评价脚本
基于test.ipynb构建，包含T5和BiLSTM两个baseline模型
"""

import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from datasets import load_from_disk
# 设置环境变量
def setup_environment():
    """设置环境变量"""
    # 设置网络代理
    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', 
                           shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value
    
    # 设置Hugging Face镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    
    print("✓ 环境变量设置完成")

# 数据加载和预处理
class DataProcessor:
    """数据处理器"""
    
    def __init__(self, train_file, valid_file=None, test_size=0.1):
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_size = test_size
        self.datasets = None
        
    def load_data(self):
        """加载数据集"""
        print("🚀 加载数据集...")
        
        # 加载训练数据
        start_time = time.time()
        train_dataset = load_dataset("json", data_files=self.train_file)
        end_time = time.time()
        print(f"✓ 训练集加载完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 加载验证数据（如果有）
        #if self.valid_file and os.path.exists(self.valid_file):
        #    valid_dataset = load_dataset("json", data_files=self.valid_file)
        #    self.datasets = {
        #        'train': train_dataset['train'],
        #        'test': valid_dataset['train']
        #    }
        #    print(f"✓ 验证集加载完成")
        #else:
        
        # 从训练集分割
        split_datasets = train_dataset["train"].train_test_split(
            test_size=self.test_size, seed=42
        )
        self.datasets = split_datasets
        print(f"✓ 数据集分割完成")
        
        print(f"训练集大小: {len(self.datasets['train'])}")
        print(f"测试集大小: {len(self.datasets['test'])}")
        
        return self.datasets

# T5模型训练器
class T5Trainer:
    """T5模型训练器"""
    
    def __init__(self, model_name="t5-small", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.metric = None
        
    def setup_model(self):
        """设置模型和分词器"""
        print(f"🔧 初始化T5模型: {self.model_name}")
        tokenizer_name = "google/mt5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.metric = evaluate.load("sacrebleu")
        
        print(f"✓ 模型初始化完成")
        print(f"词汇表大小: {len(self.tokenizer)}")
        
    def preprocess_function(self, examples):
        """预处理函数"""
        inputs = [f"translate English to Chinese: {ex}" for ex in examples["english"]]
        targets = [ex for ex in examples["chinese"]]
        
        model_inputs = self.tokenizer(
            inputs, 
            text_target=targets, 
            max_length=self.max_length, 
            truncation=True,
            padding=False
        )
        return model_inputs
    
    def preprocess_data(self, datasets, cache_path="t5_tokenized_datasets"):
        """预处理数据并缓存"""

        if os.path.exists(f"{cache_path}.train") and os.path.exists(f"{cache_path}.test"):
            print("🔄 加载已缓存的预处理数据...")
            tokenized_datasets = {
                "train": load_from_disk(f"{cache_path}.train"),
                "test": load_from_disk(f"{cache_path}.test")
            }
        else:
            print("🔄 预处理数据...")
            tokenized_datasets = datasets.map(
                self.preprocess_function,
                batched=True,
                remove_columns=datasets["train"].column_names,
                desc="Tokenizing"
            )
            print("💾 缓存预处理数据...")
            tokenized_datasets["train"].save_to_disk(f"{cache_path}.train")
            tokenized_datasets["test"].save_to_disk(f"{cache_path}.test")
        print("✓ 数据预处理完成")
        return tokenized_datasets
    
    def postprocess_text(self, preds, labels):
        """后处理文本"""
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    
    def compute_metrics(self, eval_preds):
        """计算评估指标"""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def train(self, tokenized_datasets, output_dir="t5_model", epochs=3, batch_size=16):
        """训练模型"""
        print("🚀 开始训练T5模型...")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_dir=f"{output_dir}/logs",
            logging_steps=500,
            save_steps=1000,
            eval_steps=1000,
            report_to=None,  # 禁用wandb等
        )
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # 训练模型
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ T5模型训练完成，保存在: {output_dir}")
        return trainer

# BiLSTM模型定义
class BiLSTMTranslator(nn.Module):
    """BiLSTM翻译模型"""
    
    def __init__(self, vocab_size_src, vocab_size_tgt, embed_dim=256, 
                 hidden_dim=512, num_layers=2, dropout=0.3):
        super(BiLSTMTranslator, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size_tgt = vocab_size_tgt
        
        # 编码器
        self.encoder_embedding = nn.Embedding(vocab_size_src, embed_dim)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                                   batch_first=True, bidirectional=True, dropout=dropout)
        
        # 解码器
        self.decoder_embedding = nn.Embedding(vocab_size_tgt, embed_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim * 2, num_layers, 
                                   batch_first=True, dropout=dropout)
        
        # 注意力机制
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_combine = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        
        # 输出层
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size_tgt)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt=None, max_length=50):
        """前向传播"""
        batch_size = src.size(0)
        
        # 编码器
        src_embedded = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(src_embedded)
        
        # 解码器初始化
        decoder_hidden = hidden[-self.num_layers:].contiguous()
        decoder_cell = cell[-self.num_layers:].contiguous()
        
        if tgt is not None:
            # 训练模式
            tgt_embedded = self.decoder_embedding(tgt)
            decoder_outputs = []
            
            for i in range(tgt.size(1)):
                decoder_input = tgt_embedded[:, i:i+1]
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                    decoder_input, (decoder_hidden, decoder_cell)
                )
                
                # 注意力机制
                attention_weights = torch.softmax(
                    torch.bmm(decoder_output, encoder_outputs.transpose(1, 2)), dim=2
                )
                context = torch.bmm(attention_weights, encoder_outputs)
                
                # 组合上下文和解码器输出
                combined = torch.cat([decoder_output, context], dim=2)
                combined = self.attention_combine(combined)
                
                # 输出投影
                output = self.output_projection(combined)
                decoder_outputs.append(output)
            
            return torch.cat(decoder_outputs, dim=1)
        else:
            # 推理模式
            decoder_outputs = []
            decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=src.device)
            
            for i in range(max_length):
                decoder_input_embedded = self.decoder_embedding(decoder_input)
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                    decoder_input_embedded, (decoder_hidden, decoder_cell)
                )
                
                # 注意力机制
                attention_weights = torch.softmax(
                    torch.bmm(decoder_output, encoder_outputs.transpose(1, 2)), dim=2
                )
                context = torch.bmm(attention_weights, encoder_outputs)
                
                # 组合上下文和解码器输出
                combined = torch.cat([decoder_output, context], dim=2)
                combined = self.attention_combine(combined)
                
                # 输出投影
                output = self.output_projection(combined)
                decoder_outputs.append(output)
                
                # 下一个输入
                decoder_input = output.argmax(dim=-1)
            
            return torch.cat(decoder_outputs, dim=1)

# BiLSTM训练器
class BiLSTMTrainer:
    """BiLSTM训练器"""
    
    def __init__(self, vocab_size_src=10000, vocab_size_tgt=10000, embed_dim=256, 
                 hidden_dim=512, num_layers=2, dropout=0.3):
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.model = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_vocab(self, datasets):
        """构建词汇表"""
        print("🔤 构建词汇表...")
        
        from collections import Counter
        
        # 统计词频
        src_counter = Counter()
        tgt_counter = Counter()
        
        for example in datasets['train']:
            # 简单的分词
            src_tokens = example['english'].lower().split()
            tgt_tokens = list(example['chinese'])  # 字符级分词
            
            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)
        
        # 构建词汇表
        self.src_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.tgt_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        
        # 添加高频词
        for word, freq in src_counter.most_common(self.vocab_size_src - 4):
            if freq >= 2:
                self.src_vocab[word] = len(self.src_vocab)
        
        for word, freq in tgt_counter.most_common(self.vocab_size_tgt - 4):
            if freq >= 2:
                self.tgt_vocab[word] = len(self.tgt_vocab)
        
        print(f"✓ 词汇表构建完成")
        print(f"源语言词汇表大小: {len(self.src_vocab)}")
        print(f"目标语言词汇表大小: {len(self.tgt_vocab)}")
        
        return self.src_vocab, self.tgt_vocab
    
    def setup_model(self):
        """设置模型"""
        print("🔧 初始化BiLSTM模型...")
        
        self.model = BiLSTMTranslator(
            vocab_size_src=len(self.src_vocab),
            vocab_size_tgt=len(self.tgt_vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"✓ BiLSTM模型初始化完成")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
    def preprocess_data(self, datasets):
        """预处理数据"""
        print("🔄 预处理BiLSTM数据...")
        cache_dir = "bilstm_tokenized_cache"
        cache_file = os.path.join(cache_dir, "processed_data.json")
        if os.path.exists(cache_file):
            print("🔄 加载已缓存的BiLSTM预处理数据...")
            with open(cache_file, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
                return processed_data
        
        processed_data = {'train': [], 'test': []}
        
        for split in ['train', 'test']:
            for example in datasets[split]:
                # 分词
                src_tokens = example['english'].lower().split()
                tgt_tokens = list(example['chinese'])
                
                # 转换为ID
                src_ids = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in src_tokens]
                tgt_ids = [self.tgt_vocab['<sos>']] + \
                         [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in tgt_tokens] + \
                         [self.tgt_vocab['<eos>']]
                
                # 长度限制
                if len(src_ids) <= 50 and len(tgt_ids) <= 50:
                    processed_data[split].append({
                        'src': src_ids,
                        'tgt': tgt_ids
                    })
        
        print(f"✓ BiLSTM数据预处理完成")
        print(f"训练样本数: {len(processed_data['train'])}")
        print(f"测试样本数: {len(processed_data['test'])}")
        # 保存预处理后的数据到指定位置，等到下次调用时直接加载

        os.makedirs(cache_dir, exist_ok=True)
        print("💾 缓存BiLSTM预处理数据...")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        return processed_data
    
    def create_dataloader(self, data, batch_size=32, shuffle=True):
        """创建数据加载器"""
        def collate_fn(batch):
            src_batch = []
            tgt_batch = []
            
            for item in batch:
                src_batch.append(torch.tensor(item['src'], dtype=torch.long))
                tgt_batch.append(torch.tensor(item['tgt'], dtype=torch.long))
            
            # 填充到相同长度
            from torch.nn.utils.rnn import pad_sequence

            src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.src_vocab['<pad>'])
            tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_vocab['<pad>'])
            
            return src_batch, tgt_batch
        
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    def train(self, processed_data, epochs=10, batch_size=32, lr=0.001):
        """训练模型"""
        print("🚀 开始训练BiLSTM模型...")
        
        train_loader = self.create_dataloader(processed_data['train'], batch_size, shuffle=True)
        test_loader = self.create_dataloader(processed_data['test'], batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab['<pad>'])
        
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            
            for src_batch, tgt_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                tgt_input = tgt_batch[:, :-1]  # 除最后一个token
                tgt_output = tgt_batch[:, 1:]  # 除第一个token
                
                outputs = self.model(src_batch, tgt_input)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 评估阶段
            self.model.eval()
            total_test_loss = 0
            
            with torch.no_grad():
                for src_batch, tgt_batch in test_loader:
                    src_batch = src_batch.to(self.device)
                    tgt_batch = tgt_batch.to(self.device)
                    
                    tgt_input = tgt_batch[:, :-1]
                    tgt_output = tgt_batch[:, 1:]
                    
                    outputs = self.model(src_batch, tgt_input)
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
                    
                    total_test_loss += loss.item()
            
            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}")
        
        print("✓ BiLSTM模型训练完成")
        return train_losses, test_losses

def train_lstm_model(datasets):
        # 训练BiLSTM模型
    print("\n" + "="*50)
    print("训练BiLSTM模型")
    print("="*50)
    
    bilstm_trainer = BiLSTMTrainer(
        vocab_size_src=100000,
        vocab_size_tgt=10000,
        embed_dim=256,
        hidden_dim=2048,
        num_layers=2,
        dropout=0.3
    )
    
    # 构建词汇表
    bilstm_trainer.build_vocab(datasets)
    bilstm_trainer.setup_model()
    
    # 预处理数据
    processed_data = bilstm_trainer.preprocess_data(datasets)
    
    # 使用较小的数据集
    small_processed = {
        'train': processed_data['train'][:500000],
        'test': processed_data['test'][:10000]
    }
    
    # 训练模型
    train_losses, test_losses = bilstm_trainer.train(
        small_processed, 
        epochs=5, 
        batch_size=32, 
        lr=0.001
    )
    
    # 保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BiLSTM Training Progress')
    plt.legend()
    plt.savefig('bilstm_training_loss.png')
    print("✓ 训练损失曲线已保存为 bilstm_training_loss.png")
    
    # 保存BiLSTM模型
    torch.save({
        'model_state_dict': bilstm_trainer.model.state_dict(),
        'src_vocab': bilstm_trainer.src_vocab,
        'tgt_vocab': bilstm_trainer.tgt_vocab,
        'model_config': {
            'vocab_size_src': len(bilstm_trainer.src_vocab),
            'vocab_size_tgt': len(bilstm_trainer.tgt_vocab),
            'embed_dim': bilstm_trainer.embed_dim,
            'hidden_dim': bilstm_trainer.hidden_dim,
            'num_layers': bilstm_trainer.num_layers,
            'dropout': bilstm_trainer.dropout
        }
    }, 'bilstm_translation_model.pth')
    
    print("✓ BiLSTM模型已保存为 bilstm_translation_model.pth")

def train_t5_model(datasets,model_name="t5-small",size=0.1):
     # 训练T5模型
    print("\n" + "="*50)
    print("训练T5模型")
    print("="*50)

    t5_trainer = T5Trainer(model_name=model_name, max_length=512)
    t5_trainer.setup_model()
    tokenized_datasets = t5_trainer.preprocess_data(datasets)
    
    # 使用较小的数据集进行快速训练
    small_train = tokenized_datasets["train"].select(range(int(size*len(tokenized_datasets["train"]))))
    small_test = tokenized_datasets["test"].select(range(10000))
    small_datasets = {"train": small_train, "test": small_test}
    output_dir = f"t5/{model_name}_{size}"
    t5_trainer.train(small_datasets, output_dir=output_dir, epochs=2, batch_size=256)
    
def scale_law_for_t5_model(datasets):
    train_t5_model(datasets, model_name="t5-small", size=0.001)
    #train_t5_model(datasets, model_name="t5-small", size=0.01)
    #train_t5_model(datasets, model_name="t5-small", size=0.1)
    #train_t5_model(datasets, model_name="t5-small", size=1)
    #train_t5_model(datasets, model_name="t5-base", size=0.1)
    #train_t5_model(datasets, model_name="t5-large", size=0.1)
    

# 主训练函数
def main():
    """主训练函数"""
    print("=== 英中翻译模型训练 ===")
    
    # 设置环境
    setup_environment()
    
    # 数据处理
    data_processor = DataProcessor(
        train_file="data/translation2019zh_train.json",
        valid_file="data/translation2019zh_valid.json",
        test_size=0.01
    )
    
    datasets = data_processor.load_data()
    #train_t5_model(datasets, model_name="t5-small", size=0.1)
    #train_lstm_model(datasets)
    scale_law_for_t5_model(datasets)

    
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print("模型保存位置:")
    print("- T5模型: t5_translation_model/")
    print("- BiLSTM模型: bilstm_translation_model.pth")
    print("- 训练曲线: bilstm_training_loss.png")

if __name__ == "__main__":
    main()
