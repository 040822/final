#!/usr/bin/env python3
"""
中文到英文翻译模型训练和评价脚本
使用mBART模型并绘制BLEU变化曲线
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
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    TrainerCallback
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

# 自定义回调函数来记录BLEU分数
class BleuTrackingCallback(TrainerCallback):
    """追踪训练过程中BLEU分数变化的回调函数"""
    
    def __init__(self):
        self.bleu_scores = []
        self.steps = []
        self.epochs = []
        
    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        """在每次评估后记录BLEU分数"""
        if hasattr(state, 'log_history'):
            for log in state.log_history:
                if 'eval_bleu' in log:
                    self.bleu_scores.append(log['eval_bleu'])
                    self.steps.append(log.get('step', len(self.bleu_scores)))
                    self.epochs.append(log.get('epoch', len(self.bleu_scores)))
                    break
    
    def save_bleu_curve(self, output_dir):
        """保存BLEU分数变化曲线"""
        if not self.bleu_scores:
            print("⚠️ 没有BLEU分数数据可绘制")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 绘制BLEU分数随训练步数的变化
        plt.subplot(2, 1, 1)
        plt.plot(self.steps, self.bleu_scores, 'b-o', linewidth=2, markersize=4)
        plt.title('BLEU分数随训练步数变化', fontsize=14, fontweight='bold')
        plt.xlabel('训练步数')
        plt.ylabel('BLEU分数')
        plt.grid(True, alpha=0.3)
        
        # 绘制BLEU分数随epoch的变化
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs, self.bleu_scores, 'r-s', linewidth=2, markersize=4)
        plt.title('BLEU分数随训练轮次变化', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮次')
        plt.ylabel('BLEU分数')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/bleu_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        bleu_data = {
            'steps': self.steps,
            'epochs': self.epochs,
            'bleu_scores': self.bleu_scores
        }
        with open(f"{output_dir}/bleu_data.json", 'w', encoding='utf-8') as f:
            json.dump(bleu_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ BLEU变化曲线已保存到 {output_dir}/bleu_curve.png")
        print(f"✓ BLEU数据已保存到 {output_dir}/bleu_data.json")

# mBART模型训练器
class MBartTrainer:
    """mBART模型训练器"""
    
    def __init__(self, model_name="facebook/mbart-large-50", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.metric = None
        self.bleu_callback = None
        
    def setup_model(self):
        """设置模型和分词器"""
        print(f"🔧 初始化mBART模型: {self.model_name}")
        
        # 加载mBART专用的tokenizer和model
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name,use_safetensors=True)
        
        # 设置源语言和目标语言 (cc25版本使用不同的语言代码)
        self.tokenizer.src_lang = "zh_CN" 
        self.tokenizer.tgt_lang = "en_XX"
        
        self.metric = evaluate.load("sacrebleu")
        self.bleu_callback = BleuTrackingCallback()
        
        print(f"✓ mBART模型初始化完成")
        print(f"词汇表大小: {len(self.tokenizer)}")
        print(f"源语言: {self.tokenizer.src_lang}")
        print(f"目标语言: {self.tokenizer.tgt_lang}")
        
    def preprocess_function(self, examples):
        """预处理函数"""
        # mBART不需要特殊的prompt，直接使用原文
        inputs = examples["chinese"]
        targets = examples["english"]
        
        # 设置源语言
        self.tokenizer.src_lang = "zh_CN"
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_length, 
            truncation=True,
            padding=False
        )
        
        # 设置目标语言并编码目标文本
        self.tokenizer.tgt_lang = "en_XX"
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, 
                max_length=self.max_length, 
                truncation=True,
                padding=False
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_data(self, datasets, cache_path="mbart_tokenized_datasets", use_cache=True):
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
                desc="Tokenizing for mBART"
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
        
        # 解码预测结果
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # 处理标签
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        for i in range(5):
            # 打印前5个预测和标签结果
            print(f"[{i+1}] 预测: {decoded_preds[i]}")
            print(f"    标签: {decoded_labels[i]}") 
        
        # 计算BLEU分数
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        
        
        return result
    
    def train(self, tokenized_datasets, output_dir="mbart_model", epochs=3, batch_size=64):
        """训练模型"""
        print("🚀 开始训练mBART模型...")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",  # 改为按步数评估
            eval_steps=100,  # 每100步评估一次
            learning_rate=3e-5,  # mBART推荐的学习率
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            report_to=None,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            warmup_steps=500,  # 预热步数
        )
        
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model,
            padding=True
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[self.bleu_callback],  # 添加BLEU追踪回调
        )
        
        
        evaluate_result=trainer.evaluate()  # 初始评估
        print("🔄 初始评估结果:")
        print(evaluate_result)
               
        # 打印前5行推理结果
        print("\n" + "="*50)
        print("模型推理示例")
        print("="*50)
        
        test_dataset = tokenized_datasets["test"]
        test_samples = test_dataset.select(range(5))
        batch = data_collator([test_samples[i] for i in range(len(test_samples))])

        # 获取输入和标签
        input_ids = batch["input_ids"].to(self.model.device)
        labels = batch["labels"]

        # 解码输入
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # 解码标签（处理-100）
        labels_decoded = []
        for label_seq in labels:
            label_seq = torch.where(label_seq != -100, label_seq, self.tokenizer.pad_token_id)
            decoded_label = self.tokenizer.decode(label_seq, skip_special_tokens=True)
            labels_decoded.append(decoded_label)
            
        # 生成预测
        with torch.no_grad():
            # 设置目标语言token
            generated_tokens = self.model.generate(
                input_ids,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"],
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        # 解码预测结果
        preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 打印结果
        for i, (inp, pred, label) in enumerate(zip(inputs, preds, labels_decoded)):
            print(f"[{i+1}] 输入: {inp}")
            print(f"    预测: {pred}")
            print(f"    参考: {label}")
            print()
            
        
        # 训练模型
        trainer.train()
        
        # 保存BLEU变化曲线
        self.bleu_callback.save_bleu_curve(output_dir)
        
        # 打印前5行推理结果
        print("\n" + "="*50)
        print("模型推理示例")
        print("="*50)
        
        test_dataset = tokenized_datasets["test"]
        test_samples = test_dataset.select(range(5))
        batch = data_collator([test_samples[i] for i in range(len(test_samples))])

        # 获取输入和标签
        input_ids = batch["input_ids"].to(self.model.device)
        labels = batch["labels"]

        # 解码输入
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # 解码标签（处理-100）
        labels_decoded = []
        for label_seq in labels:
            label_seq = torch.where(label_seq != -100, label_seq, self.tokenizer.pad_token_id)
            decoded_label = self.tokenizer.decode(label_seq, skip_special_tokens=True)
            labels_decoded.append(decoded_label)
            
        # 生成预测
        with torch.no_grad():
            # cc25版本使用不同的强制开始token
            generated_tokens = self.model.generate(
                input_ids,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"],
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        # 解码预测结果
        preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 打印结果
        for i, (inp, pred, label) in enumerate(zip(inputs, preds, labels_decoded)):
            print(f"[{i+1}] 输入: {inp}")
            print(f"    预测: {pred}")
            print(f"    参考: {label}")
            print()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ mBART模型训练完成，保存在: {output_dir}")
        return trainer

def train_mbart_model(datasets, model_name="facebook/mbart-large-50", size=0.1, use_cache=True):
    """训练mBART模型"""
    print("\n" + "="*50)
    print("训练mBART模型")
    print("="*50)

    mbart_trainer = MBartTrainer(model_name=model_name, max_length=512)
    mbart_trainer.setup_model()
    tokenized_datasets = mbart_trainer.preprocess_data(datasets, use_cache=use_cache)
    
    # 使用较小的数据集进行快速训练
    small_train = tokenized_datasets["train"].select(range(int(size*len(tokenized_datasets["train"]))))
    small_test = tokenized_datasets["test"].select(range(min(500, len(tokenized_datasets["test"]))))
    small_datasets = {"train": small_train, "test": small_test}
    
    model_name_clean = model_name.replace("/", "_")
    output_dir = f"mbart/{model_name_clean}_{size}"
    
    mbart_trainer.train(small_datasets, output_dir=output_dir, epochs=3, batch_size=64)
    
    return output_dir

def scale_law_for_mbart_model(datasets):
    """mBART模型的规模定律实验"""
    print("\n" + "="*50)
    print("mBART模型规模定律实验")
    print("="*50)
    
    # 不同数据集大小的实验
    sizes = [0.01, 0.1]
    results = {}
    
    for size in sizes:
        print(f"\n🔄 训练数据集大小: {size*100:.1f}%")
        output_dir = train_mbart_model(
            datasets, 
            model_name="facebook/mbart-large-50", 
            size=size, 
            use_cache=False
        )
        results[f"size_{size}"] = output_dir
    
    # 绘制规模定律图表
    plot_scaling_results(results, sizes)
    
    return results

def plot_scaling_results(results, sizes):
    """绘制规模定律结果"""
    print("\n🔄 绘制规模定律结果...")
    
    final_bleu_scores = []
    
    for size in sizes:
        result_key = f"size_{size}"
        if result_key in results:
            bleu_file = f"{results[result_key]}/bleu_data.json"
            if os.path.exists(bleu_file):
                with open(bleu_file, 'r', encoding='utf-8') as f:
                    bleu_data = json.load(f)
                    if bleu_data['bleu_scores']:
                        final_bleu_scores.append(max(bleu_data['bleu_scores']))
                    else:
                        final_bleu_scores.append(0)
            else:
                final_bleu_scores.append(0)
        else:
            final_bleu_scores.append(0)
    
    # 绘制规模定律图表
    plt.figure(figsize=(10, 6))
    plt.plot([s*100 for s in sizes], final_bleu_scores, 'bo-', linewidth=2, markersize=8)
    plt.title('mBART模型性能随数据集大小变化', fontsize=14, fontweight='bold')
    plt.xlabel('训练数据百分比 (%)')
    plt.ylabel('最佳BLEU分数')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (size, score) in enumerate(zip(sizes, final_bleu_scores)):
        plt.annotate(f'{score:.2f}', 
                    (size*100, score), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig("mbart/scaling_law_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 规模定律图表已保存到 mbart/scaling_law_results.png")

# 主训练函数
def main():
    """主训练函数"""
    print("=== 中英翻译模型训练 (mBART) ===")
    
    # 设置环境
    setup_environment()
    
    # 数据处理
    data_processor = DataProcessor(
        train_file="data/translation2019zh_train1.json",
        valid_file="data/translation2019zh_valid.json",
        test_size=0.01
    )
    
    datasets = data_processor.load_data()
    
    # 运行mBART规模定律实验
    scale_law_for_mbart_model(datasets)

    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print("模型保存位置:")
    print("- mBART模型: mbart/")
    print("- BLEU变化曲线: mbart/*/bleu_curve.png")
    print("- 规模定律结果: mbart/scaling_law_results.png")

if __name__ == "__main__":
    main()
