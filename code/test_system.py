"""
系统测试脚本
验证各个模块的基本功能
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import traceback

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试核心模块
        from smart_dataset import CachedTranslationDataset, create_smart_dataloaders
        from models import BiLSTMTranslator, TransformerTranslator, LightweightTranslator
        from utils import setup_logging, calculate_bleu_score, plot_training_curves
        print("✓ 核心模块导入成功")
        
        # 测试外部依赖
        import torch
        import numpy as np
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        try:
            from transformers import AutoTokenizer
            print("✓ Transformers库可用")
        except ImportError:
            print("⚠️  Transformers库不可用")
        
        try:
            from accelerate import Accelerator
            print("✓ Accelerate库可用")
        except ImportError:
            print("⚠️  Accelerate库不可用")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_models():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        # 测试BiLSTM
        print("测试BiLSTM模型...")
        bilstm_model = BiLSTMTranslator(
            en_vocab_size=1000,
            zh_vocab_size=1000,
            embedding_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.1,
            max_len=64
        )
        
        # 测试前向传播
        batch_size = 2
        src_len = 10
        tgt_len = 12
        
        src = torch.randint(1, 1000, (batch_size, src_len))
        tgt = torch.randint(1, 1000, (batch_size, tgt_len))
        
        output = bilstm_model(src, tgt)
        print(f"✓ BiLSTM输出形状: {output.shape}")
        
        # 测试Transformer
        print("测试Transformer模型...")
        transformer_model = TransformerTranslator(
            en_vocab_size=1000,
            zh_vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            max_len=64
        )
        
        output = transformer_model(src, tgt)
        print(f"✓ Transformer输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_dataset():
    """测试数据集加载"""
    print("\n🔍 测试数据集加载...")
    
    try:
        # 创建测试数据
        test_data = [
            {"english": "Hello world", "chinese": "你好世界"},
            {"english": "How are you", "chinese": "你好吗"},
            {"english": "Good morning", "chinese": "早上好"},
            {"english": "Thank you", "chinese": "谢谢"},
            {"english": "Good bye", "chinese": "再见"}
        ]
        
        test_data_path = "test_data.json"
        with open(test_data_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 测试数据已创建: {test_data_path}")
        
        # 测试数据集
        from smart_dataset import CachedTranslationDataset
        
        dataset = CachedTranslationDataset(
            data_path=test_data_path,
            max_len=32,
            tokenizer_type='basic',  # 使用基础分词器避免依赖问题
            cache_dir='test_cache',
            sample_ratio=1.0
        )
        
        print(f"✓ 数据集大小: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"✓ 样本形状: {[x.shape if hasattr(x, 'shape') else len(x) for x in sample]}")
        
        # 清理测试文件
        os.remove(test_data_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        traceback.print_exc()
        return False

def test_utilities():
    """测试工具函数"""
    print("\n🔍 测试工具函数...")
    
    try:
        from utils import calculate_bleu_score, count_parameters
        
        # 测试BLEU计算
        predictions = [["hello", "world"], ["good", "morning"]]
        references = [[["hello", "world"]], [["good", "morning"]]]
        
        bleu = calculate_bleu_score(predictions, references)
        print(f"✓ BLEU分数计算: {bleu:.4f}")
        
        # 测试参数统计
        model = torch.nn.Linear(10, 5)
        params = count_parameters(model)
        print(f"✓ 参数统计: {params}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        traceback.print_exc()
        return False

def test_training_pipeline():
    """测试训练流程"""
    print("\n🔍 测试训练流程...")
    
    try:
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.model_type = 'bilstm'
                self.tokenizer_type = 'basic'
                self.batch_size = 2
                self.learning_rate = 0.001
                self.epochs = 1
                self.embedding_dim = 64
                self.hidden_dim = 64
                self.num_layers = 1
                self.dropout = 0.1
                self.max_len = 32
                self.use_accelerate = False
                self.log_dir = 'test_logs'
                self.model_dir = 'test_models'
                self.cache_dir = 'test_cache'
                self.sample_ratio = 1.0
                self.num_workers = 0
                self.gradient_accumulation_steps = 1
                self.max_grad_norm = 1.0
                self.log_interval = 10
                self.early_stopping = 0
        
        config = MockConfig()
        
        # 创建目录
        Path(config.log_dir).mkdir(exist_ok=True)
        Path(config.model_dir).mkdir(exist_ok=True)
        Path(config.cache_dir).mkdir(exist_ok=True)
        
        print("✓ 训练流程配置创建成功")
        
        # 测试模型创建
        from models import BiLSTMTranslator
        
        model = BiLSTMTranslator(
            en_vocab_size=1000,
            zh_vocab_size=1000,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            max_len=config.max_len
        )
        
        print("✓ 模型创建成功")
        
        # 测试优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        print("✓ 优化器和损失函数创建成功")
        
        # 清理测试目录
        import shutil
        if os.path.exists(config.log_dir):
            shutil.rmtree(config.log_dir)
        if os.path.exists(config.model_dir):
            shutil.rmtree(config.model_dir)
        if os.path.exists(config.cache_dir):
            shutil.rmtree(config.cache_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ 训练流程测试失败: {e}")
        traceback.print_exc()
        return False

def check_system_requirements():
    """检查系统要求"""
    print("\n🔍 检查系统要求...")
    
    try:
        # 检查Python版本
        python_version = sys.version_info
        print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("⚠️  建议使用Python 3.8或更高版本")
        else:
            print("✓ Python版本满足要求")
        
        # 检查PyTorch
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，版本: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")
        
        # 检查内存
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
        except ImportError:
            print("⚠️  无法检查系统内存 (需要安装psutil)")
        
        return True
        
    except Exception as e:
        print(f"❌ 系统检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 英译中翻译系统测试")
    print("=" * 50)
    
    tests = [
        ("系统要求检查", check_system_requirements),
        ("模块导入测试", test_imports),
        ("模型创建测试", test_models),
        ("数据集测试", test_dataset),
        ("工具函数测试", test_utilities),
        ("训练流程测试", test_training_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统已准备就绪。")
        print("\n🚀 下一步:")
        print("1. 准备训练数据 (data/translation2019zh_train.json)")
        print("2. 运行快速训练: python launcher.py train bilstm")
        print("3. 查看训练日志: tensorboard --logdir logs")
    else:
        print("⚠️  部分测试失败，请检查依赖和配置。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
