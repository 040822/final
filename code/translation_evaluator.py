#!/usr/bin/env python3
"""
翻译模型评估脚本
用于评估训练好的T5和BiLSTM模型
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import json
import time
from translation_trainer import BiLSTMTranslator

class TranslationEvaluator:
    """翻译模型评估器"""
    
    def __init__(self, test_data_path="data/translation2019zh_valid.json"):
        self.test_data_path = test_data_path
        self.test_data = None
        self.bleu_metric = evaluate.load("sacrebleu")
        
    def load_test_data(self, sample_size=100):
        """加载测试数据"""
        print(f"📊 加载测试数据: {self.test_data_path}")
        
        if os.path.exists(self.test_data_path):
            dataset = load_dataset("json", data_files=self.test_data_path)
            self.test_data = dataset["train"]
        else:
            # 如果没有单独的测试文件，从训练数据中采样
            dataset = load_dataset("json", data_files="data/translation2019zh_train.json")
            self.test_data = dataset["train"].select(range(-sample_size, 0))  # 取最后100个样本
        
        # 限制测试样本数量
        if len(self.test_data) > sample_size:
            self.test_data = self.test_data.select(range(sample_size))
        
        print(f"✓ 测试数据加载完成，样本数: {len(self.test_data)}")
        return self.test_data
    
    def evaluate_t5_model(self, model_path="t5_translation_model"):
        """评估T5模型"""
        print(f"\n🔍 评估T5模型: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ T5模型路径不存在: {model_path}")
            return None
        
        try:
            # 加载模型
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            predictions = []
            references = []
            translation_times = []
            
            print("正在生成翻译...")
            for example in tqdm(self.test_data):
                english_text = example['english']
                chinese_text = example['chinese']
                
                # 准备输入
                input_text = f"translate English to Chinese: {english_text}"
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 生成翻译
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        length_penalty=0.6,
                        early_stopping=True
                    )
                end_time = time.time()
                
                # 解码结果
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                predictions.append(prediction)
                references.append([chinese_text])
                translation_times.append(end_time - start_time)
            
            # 计算BLEU分数
            bleu_result = self.bleu_metric.compute(predictions=predictions, references=references)
            
            results = {
                'model_type': 'T5',
                'bleu_score': bleu_result['score'],
                'avg_translation_time': np.mean(translation_times),
                'total_samples': len(predictions),
                'predictions': predictions[:5],  # 保存前5个预测结果
                'references': [ref[0] for ref in references[:5]]
            }
            
            print(f"✓ T5模型评估完成")
            print(f"BLEU分数: {results['bleu_score']:.4f}")
            print(f"平均翻译时间: {results['avg_translation_time']:.4f}秒")
            
            return results
            
        except Exception as e:
            print(f"❌ T5模型评估失败: {e}")
            return None
    
    def evaluate_bilstm_model(self, model_path="bilstm_translation_model.pth"):
        """评估BiLSTM模型"""
        print(f"\n🔍 评估BiLSTM模型: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ BiLSTM模型路径不存在: {model_path}")
            return None
        
        try:
            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            model_config = checkpoint['model_config']
            src_vocab = checkpoint['src_vocab']
            tgt_vocab = checkpoint['tgt_vocab']
            
            # 创建模型
            model = BiLSTMTranslator(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            # 创建反向词汇表
            src_idx2word = {idx: word for word, idx in src_vocab.items()}
            tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
            
            predictions = []
            references = []
            translation_times = []
            
            print("正在生成翻译...")
            for example in tqdm(self.test_data):
                english_text = example['english']
                chinese_text = example['chinese']
                
                # 预处理输入
                src_tokens = english_text.lower().split()
                src_ids = [src_vocab.get(token, src_vocab['<unk>']) for token in src_tokens]
                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
                
                # 生成翻译
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(src_tensor, max_length=50)
                    predicted_ids = outputs.argmax(dim=-1).squeeze().cpu().numpy()
                end_time = time.time()
                
                # 解码结果
                prediction_tokens = []
                for idx in predicted_ids:
                    if idx == tgt_vocab['<eos>']:
                        break
                    if idx in tgt_idx2word:
                        token = tgt_idx2word[idx]
                        if token not in ['<pad>', '<unk>', '<sos>']:
                            prediction_tokens.append(token)
                
                prediction = ''.join(prediction_tokens)
                
                predictions.append(prediction)
                references.append([chinese_text])
                translation_times.append(end_time - start_time)
            
            # 计算BLEU分数
            bleu_result = self.bleu_metric.compute(predictions=predictions, references=references)
            
            results = {
                'model_type': 'BiLSTM',
                'bleu_score': bleu_result['score'],
                'avg_translation_time': np.mean(translation_times),
                'total_samples': len(predictions),
                'predictions': predictions[:5],
                'references': [ref[0] for ref in references[:5]]
            }
            
            print(f"✓ BiLSTM模型评估完成")
            print(f"BLEU分数: {results['bleu_score']:.4f}")
            print(f"平均翻译时间: {results['avg_translation_time']:.4f}秒")
            
            return results
            
        except Exception as e:
            print(f"❌ BiLSTM模型评估失败: {e}")
            return None
    
    def compare_models(self, t5_results, bilstm_results):
        """比较两个模型的性能"""
        print("\n" + "="*60)
        print("模型性能比较")
        print("="*60)
        
        if t5_results is None and bilstm_results is None:
            print("❌ 没有可比较的模型结果")
            return
        
        # 创建比较表格
        print(f"{'指标':<20} {'T5模型':<15} {'BiLSTM模型':<15}")
        print("-" * 50)
        
        if t5_results and bilstm_results:
            print(f"{'BLEU分数':<20} {t5_results['bleu_score']:<15.4f} {bilstm_results['bleu_score']:<15.4f}")
            print(f"{'平均翻译时间(秒)':<20} {t5_results['avg_translation_time']:<15.4f} {bilstm_results['avg_translation_time']:<15.4f}")
            print(f"{'测试样本数':<20} {t5_results['total_samples']:<15} {bilstm_results['total_samples']:<15}")
            
            # 确定最佳模型
            if t5_results['bleu_score'] > bilstm_results['bleu_score']:
                print(f"\n🏆 最佳模型: T5 (BLEU: {t5_results['bleu_score']:.4f})")
            else:
                print(f"\n🏆 最佳模型: BiLSTM (BLEU: {bilstm_results['bleu_score']:.4f})")
        
        elif t5_results:
            print(f"{'BLEU分数':<20} {t5_results['bleu_score']:<15.4f} {'N/A':<15}")
            print(f"{'平均翻译时间(秒)':<20} {t5_results['avg_translation_time']:<15.4f} {'N/A':<15}")
            print(f"{'测试样本数':<20} {t5_results['total_samples']:<15} {'N/A':<15}")
            
        elif bilstm_results:
            print(f"{'BLEU分数':<20} {'N/A':<15} {bilstm_results['bleu_score']:<15.4f}")
            print(f"{'平均翻译时间(秒)':<20} {'N/A':<15} {bilstm_results['avg_translation_time']:<15.4f}")
            print(f"{'测试样本数':<20} {'N/A':<15} {bilstm_results['total_samples']:<15}")
        
        # 显示示例翻译
        print("\n" + "="*60)
        print("翻译示例")
        print("="*60)
        
        if t5_results:
            print("\n📝 T5模型翻译示例:")
            for i, (pred, ref) in enumerate(zip(t5_results['predictions'], t5_results['references'])):
                print(f"  {i+1}. 预测: {pred}")
                print(f"     参考: {ref}")
                print()
        
        if bilstm_results:
            print("\n📝 BiLSTM模型翻译示例:")
            for i, (pred, ref) in enumerate(zip(bilstm_results['predictions'], bilstm_results['references'])):
                print(f"  {i+1}. 预测: {pred}")
                print(f"     参考: {ref}")
                print()
    
    def save_results(self, results, filename="evaluation_results.json"):
        """保存评估结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 评估结果已保存到: {filename}")

def main():
    """主评估函数"""
    print("=== 翻译模型评估 ===")
    
    # 创建评估器
    print("🔧 初始化评估器...")
    evaluator = TranslationEvaluator()
    
    # 加载测试数据
    print("📥 加载测试数据...")
    evaluator.load_test_data(sample_size=50)  # 使用50个样本进行快速评估
    
    # 评估T5模型
    print("\n🔍 开始评估T5模型...")
    t5_results = evaluator.evaluate_t5_model("t5_translation_model")
    
    # 评估BiLSTM模型
    print("\n🔍 开始评估BiLSTM模型...")
    bilstm_results = evaluator.evaluate_bilstm_model("bilstm_translation_model.pth")
    
    # 比较模型
    print("\n📊 比较模型性能...")
    evaluator.compare_models(t5_results, bilstm_results)
    
    # 保存结果
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        't5_results': t5_results,
        'bilstm_results': bilstm_results
    }
    
    evaluator.save_results(all_results)
    
    print("\n✓ 评估完成！")

if __name__ == "__main__":
    main()
