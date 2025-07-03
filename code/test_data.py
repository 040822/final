#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的数据加载测试脚本
"""

import torch
from dataset import create_dataloaders

def test_data_loading():
    """测试数据加载功能"""
    print("=== 测试数据加载 ===")
    dir_train = r"H:\BaiduSyncdisk\Homework\DeepLearning\final\code\data\translation2019zh_train.json"
    dir_valid = r"H:\BaiduSyncdisk\Homework\DeepLearning\final\code\data\translation2019zh_valid.json"

    try:
        # 创建小批量的数据加载器进行测试
        train_loader, valid_loader, vocab_en, vocab_zh = create_dataloaders(
            train_path=dir_train,
            valid_path=dir_valid,
            batch_size=4,
            max_len=64  # 使用较小的长度进行测试
        )
        
        print(f"✓ 成功创建数据加载器")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(valid_loader)}")
        print(f"  英文词汇表大小: {len(vocab_en)}")
        print(f"  中文词汇表大小: {len(vocab_zh)}")
        
        # 测试第一个批次
        for batch in train_loader:
            print(f"  批次形状 - 英文: {batch['english'].shape}, 中文: {batch['chinese'].shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def main():
    """主函数"""
    print("神经网络翻译系统 - 数据加载测试")
    print("=" * 50)
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
    
    print("\n" + "=" * 50)
    
    # 测试数据加载
    success = test_data_loading()
    
    if success:
        print("\n✓ 数据加载测试通过！")
        print("现在可以运行完整的训练:")
        print("  python train.py")
    else:
        print("\n✗ 数据加载测试失败")
        print("请检查数据文件是否存在，并安装必要的依赖包")

if __name__ == "__main__":
    main()
