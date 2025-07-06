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
    Seq2SeqTrainer,
    T5EncoderModel
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
        if self.model_name=="t5-small":
            tokenizer_name = "google/m"+self.model_name
        else:
            tokenizer_name = self.model_name
        print(f"使用分词器: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name,use_safetensors=True)
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

    def preprocess_data(self, datasets, cache_path="t5_tokenized_datasets", use_cache=True):
        """预处理数据并缓存"""

        if os.path.exists(f"{cache_path}.train") and os.path.exists(f"{cache_path}.test") and use_cache:
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
            learning_rate=1e-6,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=1000,
            eval_steps=1000,
            report_to=None,  # 禁用wandb等
            max_grad_norm=1.0,
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
        
        #evaluate_result = trainer.evaluate()
        #print("Evaluation Results:", evaluate_result)
        # 打印前5行推理结果
        test_dataset = tokenized_datasets["test"]
        test_samples = test_dataset.select(range(5))
            # 使用data_collator进行正确的批处理
        batch = data_collator([test_samples[i] for i in range(len(test_samples))])

        # 获取输入和标签
        input_ids = batch["input_ids"].to(self.model.device)
        labels = batch["labels"]

        # 解码输入
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # 解码标签（处理-100）
        labels_decoded = []
        for label_seq in labels:
            # 将-100替换为pad_token_id
            label_seq = torch.where(label_seq != -100, label_seq, self.tokenizer.pad_token_id)
            decoded_label = self.tokenizer.decode(label_seq, skip_special_tokens=True)
            labels_decoded.append(decoded_label)
                # 生成预测
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True
            )

        # 解码预测结果
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
        
        # 打印结果
        for i, (inp, pred, label) in enumerate(zip(inputs, preds, labels_decoded)):
            print(f"[{i+1}] 输入: {inp}")
            print(f"    预测: {pred}")
            print(f"    参考: {label}")
            print()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ T5模型训练完成，保存在: {output_dir}")
        return trainer

def train_t5_model(datasets,model_name="t5-small",size=0.1,use_cache=True):
     # 训练T5模型
    print("\n" + "="*50)
    print("训练T5模型")
    print("="*50)

    t5_trainer = T5Trainer(model_name=model_name, max_length=512)
    t5_trainer.setup_model()
    tokenized_datasets = t5_trainer.preprocess_data(datasets,use_cache=use_cache)
    
    # 使用较小的数据集进行快速训练
    small_train = tokenized_datasets["train"].select(range(int(size*len(tokenized_datasets["train"]))))
    small_test = tokenized_datasets["test"].select(range(min(10000, len(tokenized_datasets["test"]))))
    small_datasets = {"train": small_train, "test": small_test}
    output_dir = f"t5/{model_name}_{size}"
    t5_trainer.train(small_datasets, output_dir=output_dir, epochs=1, batch_size=64)
    
def scale_law_for_t5_model(datasets):
    train_t5_model(datasets, model_name="google/mt5-small", size=0.1,use_cache=False)
    #train_t5_model(datasets, model_name="t5-small", size=0.01)
    #train_t5_model(datasets, model_name="t5-small", size=0.1)
    #train_t5_model(datasets, model_name="t5-small", size=1)
    #train_t5_model(datasets, model_name="t5-base", size=0.1)
    #train_t5_model(datasets, model_name="t5-large", size=0.1)
    

# T5混合模型训练器
class T5HybridModel(nn.Module):
    """T5编码器 + 线性层 + 中文tokenizer的混合模型"""
    
    def __init__(self, t5_model_name="t5-small", hidden_dim=512, max_length=128):
        super().__init__()
        self.max_length = max_length
        
        # T5编码器
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)
        self.t5_hidden_size = self.t5_encoder.config.d_model
        
        # 简化：直接使用T5的tokenizer进行中文处理
        print("🔧 使用T5 tokenizer处理中文...")
        self.zh_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        
        print(f"✓ Tokenizer设置完成，词汇表大小: {len(self.zh_tokenizer)}")
        
        # 设置特殊token ID
        self.pad_token_id = self.zh_tokenizer.pad_token_id
        self.eos_token_id = self.zh_tokenizer.eos_token_id
        
        # 修复：使用正确的开始token
        # T5使用decoder_start_token_id，如果没有则使用pad_token_id
        if hasattr(self.zh_tokenizer, 'decoder_start_token_id') and self.zh_tokenizer.decoder_start_token_id is not None:
            self.start_token_id = self.zh_tokenizer.decoder_start_token_id
        else:
            self.start_token_id = self.pad_token_id
        
        vocab_size = len(self.zh_tokenizer)
        
        # 简化的解码器架构
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.projection = nn.Linear(self.t5_hidden_size, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # 简化的解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2  # 减少层数
        )
        
        # 位置编码
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        
        # 修复：正确的权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        # 使用更小的初始化值
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def tokenize_chinese(self, texts):
        """中文文本tokenization - 使用T5 tokenizer"""
        if isinstance(texts, str):
            texts = [texts]
        
        tokenized = self.zh_tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return tokenized["input_ids"]
    
    def decode_chinese(self, token_ids):
        """解码中文token ids为文本"""
        if token_ids.dim() > 1:
            return self.zh_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        else:
            return self.zh_tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # T5编码器
        encoder_outputs = self.t5_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden = encoder_outputs.last_hidden_state
        
        # 投影编码器输出
        memory = self.projection(encoder_hidden)
        memory = self.dropout(memory)
        
        if labels is not None:
            # 训练模式
            # 准备解码器输入（teacher forcing）
            decoder_input_ids = labels[:, :-1].contiguous()  # 去掉最后一个token
            decoder_targets = labels[:, 1:].contiguous()     # 去掉第一个token
            
            # 词嵌入
            tgt_emb = self.decoder_embedding(decoder_input_ids)
            
            # 位置编码
            seq_len = decoder_input_ids.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            
            # 组合嵌入
            tgt_emb = tgt_emb + pos_emb
            tgt_emb = self.dropout(tgt_emb)
            
            # 创建因果掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            # 解码器
            memory_key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
            
            decoder_output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # 输出投影
            logits = self.output_projection(decoder_output)
            
            # 计算损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))
            
            return {
                'loss': loss,
                'logits': logits
            }
        else:
            # 推理模式
            return self.generate(memory, attention_mask)
    
    def generate(self, memory, attention_mask=None, max_length=None):
        """生成序列 - 修复生成逻辑"""
        if max_length is None:
            max_length = self.max_length
        
        batch_size = memory.size(0)
        device = memory.device
        
        # 修复：使用正确的起始token
        generated = torch.full((batch_size, 1), self.start_token_id, device=device)
        
        memory_key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        for step in range(max_length - 1):
            current_len = generated.size(1)
            
            # 词嵌入
            tgt_emb = self.decoder_embedding(generated)
            
            # 位置编码
            positions = torch.arange(current_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            
            # 组合嵌入
            tgt_emb = tgt_emb + pos_emb
            
            # 因果掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_len).to(device)
            
            # 解码器
            decoder_output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # 输出投影
            logits = self.output_projection(decoder_output)
            
            # 取最后一个时间步并添加温度控制
            next_token_logits = logits[:, -1, :] / 0.8  # 温度控制
            
            # 修复：使用top-k采样而不是贪心搜索
            top_k = 50
            if top_k > 0:
                # 保留top-k个最可能的token
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                # 对其他位置设置为负无穷
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # 使用softmax概率分布采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(1)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            
            # 检查结束条件
            if self.eos_token_id is not None and (next_tokens == self.eos_token_id).all():
                break
            
            # 避免无限循环
            if step > 0 and (next_tokens == self.pad_token_id).all():
                break
        
        return generated

class T5HybridTrainer:
    """T5混合模型训练器"""
    
    def __init__(self, t5_model_name="t5-small", max_length=128):
        self.t5_model_name = t5_model_name
        self.max_length = max_length
        self.en_tokenizer = None
        self.model = None
        self.metric = None
        
    def setup_model(self):
        """设置模型"""
        print(f"🔧 初始化T5混合模型: {self.t5_model_name}")
        
        # 英文tokenizer (T5原生)
        self.en_tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)
        
        # 混合模型
        self.model = T5HybridModel(
            t5_model_name=self.t5_model_name,
            max_length=self.max_length
        )
        
        # 评估指标
        self.metric = evaluate.load("sacrebleu")
        
        print(f"✓ 混合模型初始化完成")
    
    def preprocess_function(self, examples):
        """预处理函数 - 修复数据处理"""
        # 英文输入
        inputs = [f"translate English to Chinese: {ex}" for ex in examples["english"]]
        model_inputs = self.en_tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 中文标签 - 现在使用相同的tokenizer
        targets = examples["chinese"]
        with self.en_tokenizer.as_target_tokenizer():
            labels = self.en_tokenizer(
                targets,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def preprocess_data(self, datasets, cache_path="t5_hybrid_tokenized_v2", use_cache=True):
        """预处理数据 - 修复批处理"""
        if os.path.exists(f"{cache_path}.train") and use_cache:
            print("🔄 加载已缓存的预处理数据...")
            tokenized_datasets = {
                "train": load_from_disk(f"{cache_path}.train"),
                "test": load_from_disk(f"{cache_path}.test")
            }
        else:
            print("🔄 预处理数据...")
            
            def process_batch(examples):
                return self.preprocess_function(examples)
            
            # 修复: 正确处理datasets字典
            tokenized_datasets = {}
            for split in ["train", "test"]:
                if split in datasets:
                    tokenized_datasets[split] = datasets[split].map(
                        process_batch,
                        batched=True,
                        remove_columns=datasets[split].column_names,
                        desc=f"Tokenizing {split}"
                    )
            
            print("💾 缓存预处理数据...")
            tokenized_datasets["train"].save_to_disk(f"{cache_path}.train")
            tokenized_datasets["test"].save_to_disk(f"{cache_path}.test")
        
        print("✓ 数据预处理完成")
        return tokenized_datasets
    
    def train(self, tokenized_datasets, output_dir="t5_hybrid_model_v2", epochs=3, batch_size=8):
        """训练模型 - 修复训练循环"""
        print("🚀 开始训练T5混合模型...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        self.model.to(device)
        
        # 准备数据
        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        # 修复：使用更合适的学习率和优化器设置
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=epochs)
        
        # 训练循环
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # 移动到设备
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: loss为 {loss.item()}, 跳过此批次")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/num_batches:.4f}"
                })
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
                
                # 学习率调度
                scheduler.step()
                
                # 保存最好的模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), f"{output_dir}/best_model.pth")
                
                # 每个epoch后评估
                if epoch % 1 == 0:
                    self.evaluate(test_loader, device)
            else:
                print(f"Epoch {epoch+1} - 没有有效的训练批次")
        
        print(f"✓ T5混合模型训练完成，最佳模型保存在: {output_dir}/best_model.pth")

    def collate_fn(self, batch):
        """数据整理函数"""
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
        labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def evaluate(self, test_loader, device):
        """评估模型 - 修复评估逻辑"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        sample_predictions = []
        sample_references = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # 计算损失
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs["loss"].item()
                num_batches += 1
                
                # 收集一些样本进行展示
                if i < 3:  # 只取前3个batch的样本
                    # 生成预测
                    generated = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # 解码预测和参考
                    preds = self.model.decode_chinese(generated[:min(2, generated.size(0))])
                    refs = self.model.decode_chinese(labels[:min(2, labels.size(0))])
                    
                    # 解码输入
                    inputs = self.en_tokenizer.batch_decode(input_ids[:min(2, input_ids.size(0))], skip_special_tokens=True)
                    
                    for j, (inp, pred, ref) in enumerate(zip(inputs, preds, refs)):
                        if len(sample_predictions) < 5:  # 限制样本数量
                            sample_predictions.append(pred)
                            sample_references.append(ref)
                            print(f"样本 {len(sample_predictions)}:")
                            print(f"  输入: {inp}")
                            print(f"  预测: {pred}")
                            print(f"  参考: {ref}")
                            print()
        
        avg_loss = total_loss / num_batches
        print(f"验证损失: {avg_loss:.4f}")
        
        # 如果有足够的样本，计算BLEU分数
        if len(sample_predictions) > 0:
            try:
                refs_formatted = [[ref] for ref in sample_references]
                result = self.metric.compute(predictions=sample_predictions, references=refs_formatted)
                print(f"BLEU Score (样本): {result['score']:.4f}")
            except Exception as e:
                print(f"BLEU计算失败: {e}")
        
        self.model.train()

def train_t5_hybrid_model(datasets, size=0.001):  # 使用更小的数据集进行调试
    """训练T5混合模型"""
    print("\n" + "="*50)
    print("训练T5混合模型 (修复版 v2)")
    print("="*50)
    
    trainer = T5HybridTrainer(t5_model_name="t5-small", max_length=32)  # 进一步减少序列长度
    trainer.setup_model()
    
    # 使用更小的数据集进行调试
    small_train = datasets["train"].select(range(int(size * len(datasets["train"]))))
    small_test = datasets["test"].select(range(min(100, len(datasets["test"]))))
    small_datasets = {"train": small_train, "test": small_test}
    
    print(f"训练集大小: {len(small_train)}")
    print(f"测试集大小: {len(small_test)}")
    
    tokenized_datasets = trainer.preprocess_data(small_datasets, use_cache=False)
    trainer.train(tokenized_datasets, output_dir="t5_hybrid_model_fixed_v2", epochs=5, batch_size=2)

# 主训练函数
def main():
    """主训练函数"""
    print("=== 英中翻译模型训练 ===")
    
    # 设置环境
    setup_environment()
    
    # 数据处理
    data_processor = DataProcessor(
        train_file="data/translation2019zh_train1.json",
        valid_file="data/translation2019zh_valid.json",
        test_size=0.01
    )
    
    datasets = data_processor.load_data()
    #train_t5_model(datasets, model_name="t5-small", size=0.1)
    #train_lstm_model(datasets)
    #scale_law_for_t5_model(datasets)
    train_t5_hybrid_model(datasets, size=0.01)

    
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print("模型保存位置:")
    print("- T5模型: t5_translation_model/")
    print("- BiLSTM模型: bilstm_translation_model.pth")
    print("- 训练曲线: bilstm_training_loss.png")

    

# 主训练函数
def main():
    """主训练函数"""
    print("=== 英中翻译模型训练 ===")
    
    # 设置环境
    setup_environment()
    
    # 数据处理
    data_processor = DataProcessor(
        train_file="data/translation2019zh_train1.json",
        valid_file="data/translation2019zh_valid.json",
        test_size=0.01
    )
    
    datasets = data_processor.load_data()
    #train_t5_model(datasets, model_name="t5-small", size=0.1)
    #train_lstm_model(datasets)
    #scale_law_for_t5_model(datasets)
    train_t5_hybrid_model(datasets, size=0.01)

    
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print("模型保存位置:")
    print("- T5模型: t5_translation_model/")
    print("- BiLSTM模型: bilstm_translation_model.pth")
    print("- 训练曲线: bilstm_training_loss.png")

if __name__ == "__main__":
    main()
