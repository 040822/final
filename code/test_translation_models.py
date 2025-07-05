#!/usr/bin/env python3
"""
简化的翻译模型测试脚本
用于快速验证T5和BiLSTM模型的基本功能
"""

import os
import subprocess
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
from tqdm import tqdm
import time

def setup_environment():
    """设置环境"""
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
    print("✓ 环境设置完成")

def test_data_loading():
    """测试数据加载"""
    print("📊 测试数据加载...")
    
    try:
        # 加载训练数据
        start_time = time.time()
        train_dataset = load_dataset("json", data_files="data/translation2019zh_train.json")
        end_time = time.time()
        print(f"✓ 训练集加载完成，耗时: {end_time - start_time:.2f}秒")
        
        # 检查数据格式
        sample = train_dataset["train"][0]
        print(f"✓ 数据样本: {sample}")
        
        # 分割数据集
        split_datasets = train_dataset["train"].train_test_split(test_size=0.1, seed=42)
        print(f"✓ 训练集大小: {len(split_datasets['train'])}")
        print(f"✓ 测试集大小: {len(split_datasets['test'])}")
        
        return split_datasets
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def test_t5_model():
    """测试T5模型"""
    print("\n🔍 测试T5模型...")
    
    try:
        # 加载模型
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print(f"✓ T5模型加载成功: {model_name}")
        print(f"✓ 词汇表大小: {len(tokenizer)}")
        
        # 测试预处理
        test_input = "Hello, how are you?"
        test_target = "你好，你好吗？"
        
        inputs = tokenizer(
            f"translate English to Chinese: {test_input}",
            text_target=test_target,
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        
        print(f"✓ 预处理测试成功")
        print(f"  输入形状: {inputs['input_ids'].shape}")
        print(f"  标签形状: {inputs['labels'].shape}")
        
        # 测试推理
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=50,
                num_beams=2,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 推理测试成功")
        print(f"  输入: {test_input}")
        print(f"  输出: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ T5模型测试失败: {e}")
        return False

def test_quick_training():
    """测试快速训练"""
    print("\n🚀 测试快速训练...")
    
    try:
        # 加载数据
        datasets = test_data_loading()
        if datasets is None:
            return False
        
        # 使用很小的数据集进行测试
        tiny_train = datasets["train"].select(range(10))
        tiny_test = datasets["test"].select(range(5))
        
        # 设置模型
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # 预处理数据
        def preprocess_function(examples):
            inputs = [f"translate English to Chinese: {ex}" for ex in examples["english"]]
            targets = [ex for ex in examples["chinese"]]
            model_inputs = tokenizer(
                inputs, text_target=targets, max_length=128, truncation=True
            )
            return model_inputs
        
        tokenized_train = tiny_train.map(
            preprocess_function,
            batched=True,
            remove_columns=tiny_train.column_names,
        )
        
        tokenized_test = tiny_test.map(
            preprocess_function,
            batched=True,
            remove_columns=tiny_test.column_names,
        )
        
        print(f"✓ 数据预处理完成")
        print(f"  训练样本: {len(tokenized_train)}")
        print(f"  测试样本: {len(tokenized_test)}")
        
        # 设置训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir="./test_model",
            eval_strategy="no",  # 不进行评估以节省时间
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,  # 只训练1个epoch
            weight_decay=0.01,
            save_total_limit=1,
            predict_with_generate=True,
            logging_steps=1,
            save_steps=1000,
            report_to=None,
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # 创建训练器
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        
        # 开始训练
        print("开始快速训练...")
        trainer.train()
        
        print("✓ 快速训练完成")
        
        #测试训练后的模型
        test_input = "Good morning!"
        inputs = tokenizer(
            f"translate English to Chinese: {test_input}",
            return_tensors="pt"
        )
        
        
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            num_beams=2,
            early_stopping=True
        )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 训练后推理测试:")
        print(f"  输入: {test_input}")
        print(f"  输出: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation():
    """测试评估功能"""
    print("\n📊 测试评估功能...")
    
    try:
        # 加载BLEU评估器
        metric = evaluate.load("sacrebleu")
        
        # 测试数据
        predictions = ["你好，世界！", "这是一个测试。"]
        references = [["你好，世界！"], ["这是一个测试。"]]
        
        # 计算BLEU分数
        result = metric.compute(predictions=predictions, references=references)
        print(f"✓ BLEU评估测试成功")
        print(f"  BLEU分数: {result['score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 翻译模型功能测试 ===")
    
    # 设置环境
    setup_environment()
    
    # 测试各个组件
    tests = [
        ("数据加载", test_data_loading),
        ("T5模型", test_t5_model),
        ("评估功能", test_evaluation),
        ("快速训练", test_quick_training),
    ]
    tests = [
        ("快速训练", test_quick_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
                
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
            results[test_name] = False
    
    # 总结
    print(f"\n{'='*50}")
    print("测试总结")
    print(f"{'='*50}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 所有测试通过！可以运行完整的训练脚本。")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查相关问题。")

if __name__ == "__main__":
    main()
