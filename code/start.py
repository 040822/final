#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
英译中翻译系统 - 快速启动脚本
提供多种训练选项，从轻量级到完整版本
"""

import os
import sys
import torch

def check_requirements():
    """检查环境要求"""
    print("🔍 检查环境要求...")
    
    # 检查PyTorch
    print(f"✓ PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: GPU {torch.cuda.get_device_name()}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    else:
        print("⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
    
    # 数据加载
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)  # 确保当前工作目录是脚本所在
    print("Current working directory:", os.getcwd())
    print("root_dir:", root_dir)
    train_path = os.path.join(root_dir, "data/translation2019zh_train.json")
    valid_path = os.path.join(root_dir, "data/translation2019zh_valid.json")
    print(f"Train path: {train_path}")
    print(f"Valid path: {valid_path}")
    # 检查数据文件
    train_file = train_path
    valid_file = valid_path

    if os.path.exists(train_file) and os.path.exists(valid_file):
        print("✓ 数据文件存在")
    else:
        print("❌ 数据文件不存在，请确保数据文件在data/目录下")
        return False
    
    return True

def show_training_options():
    """显示训练选项"""
    print("\n" + "="*50)
    print("🚀 英译中神经网络翻译系统")
    print("="*50)
    print("请选择训练方式：")
    print()
    print("1. 🏃 快速测试模式 (推荐初学者)")
    print("   - 轻量级模型 (~5M参数)")
    print("   - 短序列长度 (32 tokens)")
    print("   - 8个训练轮次")
    print("   - 预计训练时间: 30分钟-1小时")
    print()
    print("2. ⚡ 轻量级模式")
    print("   - 中等模型 (~15M参数)")
    print("   - 中等序列长度 (64 tokens)")
    print("   - 10个训练轮次")
    print("   - 预计训练时间: 1-3小时")
    print()
    print("3. 🔥 完整模式")
    print("   - 大型Transformer模型 (~65M参数)")
    print("   - 完整序列长度 (128 tokens)")
    print("   - 20个训练轮次")
    print("   - 预计训练时间: 8-12小时")
    print()
    print("4. 🤖 使用预训练模型 (需要网络连接)")
    print("   - 基于BERT的翻译模型")
    print("   - 更好的初始性能")
    print("   - 5个训练轮次")
    print("   - 预计训练时间: 2-4小时")
    print()
    print("5. 📊 仅测试数据加载")
    print("   - 验证数据处理是否正常")
    print("   - 不进行训练")
    print()

def run_quick_test():
    """快速测试模式"""
    print("\n🏃 启动快速测试模式...")
    
    # 修改train_lightweight.py的参数
    import train_lightweight
    
    # 临时修改配置
    original_main = train_lightweight.main
    
    def quick_main():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        
        from dataset import create_dataloaders
        from train_lightweight import SimplePretrainedTranslator, LightweightTrainer
        
        # 超快速配置
        train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
            train_path="data/translation2019zh_train.json",
            valid_path="data/translation2019zh_valid.json",
            batch_size=8,
            max_len=16  # 非常短的序列
        )
        
        # 超轻量模型
        model = SimplePretrainedTranslator(
            src_vocab_size=len(vocab_en),
            tgt_vocab_size=len(vocab_zh),
            embedding_dim=128,  # 更小
            hidden_dim=256,     # 更小
            num_layers=1,       # 只有1层
            dropout=0.1
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Quick test model has {total_params:,} parameters')
        
        trainer = LightweightTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            vocab_en=vocab_en,
            vocab_zh=vocab_zh,
            lr=2e-3,
            device=device,
            save_dir='checkpoints_quick'
        )
        
        trainer.train(num_epochs=3)  # 只训练3轮
    
    quick_main()

def run_lightweight():
    """轻量级模式"""
    print("\n⚡ 启动轻量级模式...")
    import train_lightweight
    train_lightweight.main()

def run_full_training():
    """完整训练模式"""
    print("\n🔥 启动完整训练模式...")
    import train
    train.main()

def run_pretrained():
    """预训练模型模式"""
    print("\n🤖 启动预训练模型模式...")
    try:
        import train_pretrained
        train_pretrained.main()
    except ImportError:
        print("❌ 缺少transformers库，请运行: pip install transformers")
        print("或选择其他训练模式")

def test_data_loading():
    """测试数据加载"""
    print("\n📊 测试数据加载...")
    import test_data
    test_data.main()

def main():
    """主函数"""
    # 检查环境
    if not check_requirements():
        print("❌ 环境检查失败，请解决上述问题后重试")
        return
    
    # 显示选项
    show_training_options()
    
    # 获取用户选择
    while True:
        try:
            #choice = input("请输入选择 (1-5): ").strip()
            choice = 4
            if choice == '1':
                run_quick_test()
                break
            elif choice == '2':
                run_lightweight()
                break
            elif choice == '3':
                run_full_training()
                break
            elif choice == '4':
                run_pretrained()
                break
            elif choice == '5':
                test_data_loading()
                break
            else:
                print("❌ 无效选择，请输入1-5")
        except KeyboardInterrupt:
            print("\n\n👋 退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            break
    
    print("\n✨ 训练完成！")
    print("📁 模型文件保存在 checkpoints/ 目录")
    print("📊 训练日志可在 TensorBoard 中查看: tensorboard --logdir=logs")

if __name__ == "__main__":
    main()
