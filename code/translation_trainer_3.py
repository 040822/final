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
