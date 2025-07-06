#!/usr/bin/env python3
"""
英文到中文翻译模型训练和评价脚本
基于test.ipynb构建，包含Qwen2.5和BiLSTM两个baseline模型
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
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
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

# Qwen模型训练器
class QwenTrainer:
    """Qwen模型训练器"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.metric = None
        
    def setup_model(self):
        """设置模型和分词器"""
        print(f"🔧 初始化Qwen模型: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.metric = evaluate.load("sacrebleu")
        
        print(f"✓ 模型初始化完成")
        print(f"词汇表大小: {len(self.tokenizer)}")
        
    def preprocess_function(self, examples):
        """预处理函数 - 为因果语言模型格式化"""
        # 构建翻译格式的输入
        texts = []
        for english, chinese in zip(examples["english"], examples["chinese"]):
            # 使用特殊格式来标识翻译任务
            text = f"Translate English to Chinese:\nEnglish: {english}\nChinese: {chinese}{self.tokenizer.eos_token}"
            texts.append(text)
        
        # 分词 - 添加padding以确保batch中所有序列长度一致
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",  # 使用max_length padding确保一致的长度
            return_tensors=None
        )
        
        # 为因果语言模型设置labels，确保数据类型正确
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        
        return tokenized

    def preprocess_data(self, datasets, cache_path="qwen_tokenized_datasets", use_cache=True):
        """预处理数据并缓存"""
        cache_train_path = f"{cache_path}.train"
        cache_test_path = f"{cache_path}.test"
        
        if os.path.exists(cache_train_path) and os.path.exists(cache_test_path) and use_cache:
            print("🔄 加载已缓存的预处理数据...")
            tokenized_datasets = {
                "train": load_from_disk(cache_train_path),
                "test": load_from_disk(cache_test_path)
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
            tokenized_datasets["train"].save_to_disk(cache_train_path)
            tokenized_datasets["test"].save_to_disk(cache_test_path)
        print("✓ 数据预处理完成")
        return tokenized_datasets
    
    def extract_translation(self, text):
        """从生成的文本中提取翻译结果"""
        # 查找"Chinese:"后的内容
        if "Chinese:" in text:
            translation = text.split("Chinese:")[-1].strip()
            # 移除可能的eos_token
            translation = translation.replace(self.tokenizer.eos_token, "").strip()
            return translation
        return text.strip()
    
    def compute_metrics(self, eval_preds):
        """计算评估指标"""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # 解码预测结果
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # 解码标签
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 提取翻译部分
        extracted_preds = [self.extract_translation(pred) for pred in decoded_preds]
        extracted_labels = [self.extract_translation(label) for label in decoded_labels]
        
        # 格式化为BLEU评估所需的格式
        formatted_labels = [[label] for label in extracted_labels]
        
        try:
            result = self.metric.compute(predictions=extracted_preds, references=formatted_labels)
            result = {"bleu": result["score"]}
        except:
            result = {"bleu": 0.0}
        
        prediction_lens = [len(pred.split()) for pred in extracted_preds]
        result["gen_len"] = np.mean(prediction_lens) if prediction_lens else 0
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def train(self, tokenized_datasets, output_dir="qwen_model", epochs=3, batch_size=16):
        """训练模型"""
        print("🚀 开始训练Qwen模型...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            fp16=False,  # 禁用FP16避免梯度缩放问题
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # 使用BF16如果支持
            logging_dir=f"{output_dir}/logs",
            logging_steps=500,
            save_steps=1000,
            eval_steps=1000,
            report_to=None,
            dataloader_drop_last=True,
            gradient_checkpointing=True,  # 启用梯度检查点节省内存
            max_grad_norm=1.0,  # 添加梯度裁剪
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 因果语言模型
            pad_to_multiple_of=None,  # 禁用pad_to_multiple_of避免长度问题
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            processing_class=self.tokenizer,  # 修复弃用警告
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # 训练模型
        trainer.train()
        
        evaluate_result = trainer.evaluate()
        print("Evaluation Results:", evaluate_result)
        
        # 打印前5行推理结果
        self.show_inference_examples(tokenized_datasets["test"])
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Qwen模型训练完成，保存在: {output_dir}")
        return trainer
    
    def show_inference_examples(self, test_dataset, num_examples=5):
        """显示推理示例"""
        print("\n=== 推理示例 ===")
        test_samples = test_dataset.select(range(min(num_examples, len(test_dataset))))
        
        for i in range(len(test_samples)):
            sample = test_samples[i]
            input_ids = torch.tensor([sample["input_ids"]]).to(self.model.device)
            
            # 找到"Chinese:"的位置，只使用到该位置的输入
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if "Chinese:" in input_text:
                prompt = input_text.split("Chinese:")[0] + "Chinese:"
                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            else:
                prompt_ids = input_ids
            
            # 生成翻译
            with torch.no_grad():
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码结果
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            original_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # 提取各部分
            if "English:" in generated_text and "Chinese:" in generated_text:
                parts = generated_text.split("Chinese:")
                english_part = parts[0].replace("Translate English to Chinese:\nEnglish:", "").strip()
                chinese_part = self.extract_translation(generated_text)
            else:
                english_part = "解析失败"
                chinese_part = "解析失败"
            
            # 提取参考翻译
            ref_chinese = self.extract_translation(original_text)
            
            print(f"[{i+1}] 英文: {english_part}")
            print(f"    预测: {chinese_part}")
            print(f"    参考: {ref_chinese}")
            print()


def train_qwen_model(datasets, model_name="Qwen/Qwen2.5-0.5B", size=0.1, use_cache=True):
    """训练Qwen模型"""
    print("\n" + "="*50)
    print("训练Qwen模型")
    print("="*50)

    qwen_trainer = QwenTrainer(model_name=model_name, max_length=256)
    qwen_trainer.setup_model()
    tokenized_datasets = qwen_trainer.preprocess_data(datasets, use_cache=use_cache)
    
    # 使用较小的数据集进行快速训练
    small_train = tokenized_datasets["train"].select(range(int(size*len(tokenized_datasets["train"]))))
    small_test = tokenized_datasets["test"].select(range(min(500, len(tokenized_datasets["test"]))))
    small_datasets = {"train": small_train, "test": small_test}
    
    # 生成安全的文件夹名称
    safe_model_name = model_name.replace("/", "_").replace(".", "_")
    output_dir = f"qwen/{safe_model_name}_{size}"
    qwen_trainer.train(small_datasets, output_dir=output_dir, epochs=1, batch_size=2)

def scale_law_for_qwen_model(datasets):
    """测试Qwen模型的缩放定律"""
    train_qwen_model(datasets, model_name="Qwen/Qwen2.5-0.5B", size=0.001, use_cache=True)
    # train_qwen_model(datasets, model_name="Qwen/Qwen2.5-0.5B", size=0.01, use_cache=True)
    # train_qwen_model(datasets, model_name="Qwen/Qwen2.5-0.5B", size=0.1, use_cache=True)

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
    # train_qwen_model(datasets, model_name="Qwen/Qwen2.5-0.5B", size=0.1)
    scale_law_for_qwen_model(datasets)

    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print("模型保存位置:")
    print("- Qwen模型: qwen/")
    print("- BiLSTM模型: bilstm_translation_model.pth")
    print("- 训练曲线: bilstm_training_loss.png")

if __name__ == "__main__":
    main()
