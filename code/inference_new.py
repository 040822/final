"""
推理和评估脚本 - 重新设计版本
支持多模型推理、批量翻译、性能评估和对比分析
"""
import argparse
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 导入本地模块
from models import (
    BiLSTMTranslator, 
    TransformerTranslator, 
    LightweightTranslator,
    PretrainedTranslator,
    create_model
)
from smart_dataset import create_smart_dataloaders, CachedTranslationDataset
from utils import (
    load_checkpoint,
    calculate_bleu_score,
    evaluate_model,
    count_parameters,
    format_time,
    get_model_size
)

class TranslationInference:
    """翻译推理器"""
    
    def __init__(self, model_path, config_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 从检查点中加载配置
            checkpoint = torch.load(model_path, map_location='cpu')
            self.config = checkpoint.get('config', {})
        
        # 加载模型
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        print(f"✓ 推理器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  模型类型: {self.config.get('model_type', 'unknown')}")
        print(f"  参数数量: {count_parameters(self.model)['total']:,}")
    
    def _load_model(self):
        """加载模型"""
        # 根据配置创建模型
        model_type = self.config.get('model_type', 'bilstm')
        
        if model_type == 'bilstm':
            model = BiLSTMTranslator(
                en_vocab_size=self.config.get('en_vocab_size', 50000),
                zh_vocab_size=self.config.get('zh_vocab_size', 50000),
                embedding_dim=self.config.get('embedding_dim', 512),
                hidden_dim=self.config.get('hidden_dim', 512),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.1),
                max_len=self.config.get('max_len', 128)
            )
        elif model_type == 'transformer':
            model = TransformerTranslator(
                en_vocab_size=self.config.get('en_vocab_size', 50000),
                zh_vocab_size=self.config.get('zh_vocab_size', 50000),
                d_model=self.config.get('d_model', 512),
                nhead=self.config.get('nhead', 8),
                num_layers=self.config.get('num_layers', 6),
                dropout=self.config.get('dropout', 0.1),
                max_len=self.config.get('max_len', 128)
            )
        elif model_type == 'lightweight':
            model = LightweightTranslator(
                tokenizer_name=self.config.get('tokenizer_name', 'bert-base-multilingual-cased'),
                max_len=self.config.get('max_len', 128),
                hidden_dim=self.config.get('hidden_dim', 512),
                dropout=self.config.get('dropout', 0.1)
            )
        elif model_type == 'pretrained':
            model = PretrainedTranslator(
                model_name=self.config.get('pretrained_model_name', 'Helsinki-NLP/opus-mt-en-zh'),
                max_len=self.config.get('max_len', 128),
                dropout=self.config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_tokenizer(self):
        """加载分词器"""
        tokenizer_type = self.config.get('tokenizer_type', 'transformers')
        tokenizer_name = self.config.get('tokenizer_name', 'bert-base-multilingual-cased')
        
        if tokenizer_type == 'transformers' and TRANSFORMERS_AVAILABLE:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            return None
    
    def translate_text(self, text, max_length=None, num_beams=4):
        """翻译单个文本"""
        if max_length is None:
            max_length = self.config.get('max_len', 128)
        
        # 预处理文本
        if self.tokenizer:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )
            src_ids = inputs['input_ids'].to(self.device)
            src_mask = inputs['attention_mask'].to(self.device)
        else:
            # 简单的文本处理（需要根据实际分词器实现）
            tokens = text.split()
            src_ids = torch.tensor([1] + [hash(token) % 50000 for token in tokens] + [2]).unsqueeze(0).to(self.device)
            src_mask = torch.ones_like(src_ids)
        
        # 生成翻译
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                # 预训练模型
                outputs = self.model.generate(
                    src_ids, src_mask,
                    max_length=max_length,
                    num_beams=num_beams
                )
                
                if self.tokenizer:
                    translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    translation = ' '.join(map(str, outputs[0].cpu().numpy()))
            else:
                # 其他模型 - 贪心解码
                batch_size = src_ids.size(0)
                generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
                generated[:, 0] = 1  # BOS token
                
                for pos in range(1, max_length):
                    if self.config.get('model_type') in ['bilstm', 'transformer']:
                        outputs = self.model(src_ids, generated[:, :pos])
                    else:
                        outputs = self.model(src_ids, src_mask, generated[:, :pos])
                    
                    next_token = outputs[:, -1, :].argmax(dim=-1)
                    generated[:, pos] = next_token
                    
                    # 检查是否生成了EOS token
                    if next_token.item() == 2:  # 假设2是EOS token
                        break
                
                if self.tokenizer:
                    translation = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                else:
                    translation = ' '.join(map(str, generated[0].cpu().numpy()))
        
        return translation
    
    def translate_batch(self, texts, batch_size=32, max_length=None):
        """批量翻译"""
        if max_length is None:
            max_length = self.config.get('max_len', 128)
        
        translations = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"开始批量翻译，共 {len(texts)} 条文本，{total_batches} 个批次")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_translations = []
            
            start_time = time.time()
            
            for text in batch_texts:
                translation = self.translate_text(text, max_length)
                batch_translations.append(translation)
            
            translations.extend(batch_translations)
            
            # 显示进度
            batch_num = i // batch_size + 1
            elapsed = time.time() - start_time
            print(f"批次 {batch_num}/{total_batches} 完成，耗时 {elapsed:.2f}s")
        
        return translations
    
    def evaluate_on_dataset(self, data_path, batch_size=32, max_samples=None):
        """在数据集上评估模型"""
        print(f"在数据集上评估模型: {data_path}")
        
        # 创建数据集
        dataset = CachedTranslationDataset(
            data_path=data_path,
            max_len=self.config.get('max_len', 128),
            tokenizer_type=self.config.get('tokenizer_type', 'transformers'),
            tokenizer_name=self.config.get('tokenizer_name', 'bert-base-multilingual-cased'),
            cache_dir=self.config.get('cache_dir', 'cache'),
            sample_ratio=1.0 if max_samples is None else max_samples / 100000  # 假设数据集大小
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=dataset.collate_fn
        )
        
        # 评估
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        results = evaluate_model(
            self.model, dataloader, criterion, self.device,
            tokenizer=self.tokenizer,
            max_batches=max_samples // batch_size if max_samples else None
        )
        
        print(f"评估结果:")
        print(f"  损失: {results['loss']:.4f}")
        print(f"  BLEU分数: {results['bleu']:.4f}")
        
        return results

class ModelComparator:
    """模型对比器"""
    
    def __init__(self, model_configs):
        """
        初始化模型对比器
        Args:
            model_configs: 模型配置列表，每个配置包含 name, model_path, config_path
        """
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for config in model_configs:
            print(f"加载模型: {config['name']}")
            self.models[config['name']] = TranslationInference(
                config['model_path'], 
                config.get('config_path'),
                self.device
            )
    
    def compare_translations(self, texts, save_path=None):
        """对比多个模型的翻译结果"""
        results = {}
        
        print(f"对比 {len(self.models)} 个模型的翻译结果...")
        
        for model_name, inference in self.models.items():
            print(f"使用模型 {model_name} 翻译...")
            start_time = time.time()
            translations = inference.translate_batch(texts)
            elapsed = time.time() - start_time
            
            results[model_name] = {
                'translations': translations,
                'time': elapsed,
                'speed': len(texts) / elapsed if elapsed > 0 else 0
            }
            
            print(f"  完成，耗时 {format_time(elapsed)}，速度 {results[model_name]['speed']:.2f} 文本/秒")
        
        # 创建对比表格
        comparison_data = []
        for i, text in enumerate(texts):
            row = {'input': text}
            for model_name in self.models.keys():
                row[f'{model_name}_output'] = results[model_name]['translations'][i]
            comparison_data.append(row)
        
        # 保存对比结果
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'comparison_data': comparison_data,
                    'performance_summary': {
                        model_name: {
                            'total_time': res['time'],
                            'avg_speed': res['speed']
                        }
                        for model_name, res in results.items()
                    }
                }, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 对比结果已保存: {save_path}")
        
        return comparison_data, results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='翻译模型推理和评估')
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 单个文本翻译
    translate_parser = subparsers.add_parser('translate', help='翻译单个文本')
    translate_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    translate_parser.add_argument('--config_path', type=str, help='配置文件路径')
    translate_parser.add_argument('--text', type=str, required=True, help='要翻译的文本')
    translate_parser.add_argument('--max_length', type=int, default=128, help='最大长度')
    translate_parser.add_argument('--num_beams', type=int, default=4, help='束搜索宽度')
    
    # 批量翻译
    batch_parser = subparsers.add_parser('batch', help='批量翻译')
    batch_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    batch_parser.add_argument('--config_path', type=str, help='配置文件路径')
    batch_parser.add_argument('--input_file', type=str, required=True, help='输入文件')
    batch_parser.add_argument('--output_file', type=str, required=True, help='输出文件')
    batch_parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    batch_parser.add_argument('--max_length', type=int, default=128, help='最大长度')
    
    # 模型评估
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    eval_parser.add_argument('--config_path', type=str, help='配置文件路径')
    eval_parser.add_argument('--data_path', type=str, required=True, help='测试数据路径')
    eval_parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    eval_parser.add_argument('--max_samples', type=int, default=1000, help='最大样本数')
    eval_parser.add_argument('--output_file', type=str, help='结果输出文件')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.command == 'translate':
        # 单个文本翻译
        inference = TranslationInference(args.model_path, args.config_path)
        translation = inference.translate_text(
            args.text, 
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        print(f"原文: {args.text}")
        print(f"译文: {translation}")
    
    elif args.command == 'batch':
        # 批量翻译
        inference = TranslationInference(args.model_path, args.config_path)
        
        # 读取输入文件
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.json'):
                data = json.load(f)
                texts = [item['english'] for item in data]
            else:
                texts = [line.strip() for line in f if line.strip()]
        
        # 批量翻译
        translations = inference.translate_batch(
            texts, 
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # 保存结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for original, translation in zip(texts, translations):
                f.write(f"{original}\t{translation}\n")
        
        print(f"✓ 翻译完成，结果已保存: {args.output_file}")
    
    elif args.command == 'evaluate':
        # 模型评估
        inference = TranslationInference(args.model_path, args.config_path)
        results = inference.evaluate_on_dataset(
            args.data_path,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ 评估结果已保存: {args.output_file}")
    
    else:
        print("请指定有效的命令: translate, batch, evaluate")

if __name__ == "__main__":
    main()
