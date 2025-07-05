# 英译中神经网络翻译系统

一个高效、可扩展的英文到中文神经网络翻译系统，支持多种模型架构、多卡训练、智能数据加载和完整的推理评估流程。

## 🚀 特性

### 核心功能
- **多模型架构支持**: BiLSTM、Transformer、轻量级模型、预训练模型
- **高效数据处理**: 智能缓存、并行加载、自动数据分割
- **多卡训练**: 集成Accelerate库，支持分布式训练
- **灵活分词**: 支持Transformers和SpaCy分词器
- **完整推理**: 单文本翻译、批量处理、模型对比
- **可视化训练**: 实时训练曲线、损失监控、性能分析

### 技术亮点
- **智能缓存系统**: 避免重复预处理，大幅提升数据加载速度
- **内存优化**: 支持大数据集的分块处理和采样
- **多进程并行**: 充分利用多核CPU进行数据预处理
- **混合精度训练**: 减少显存占用，提升训练速度
- **自动早停**: 防止过拟合，节省训练时间

## 📋 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (可选，用于GPU训练)
- 8GB+ RAM (推荐16GB+)
- 4GB+ 显存 (用于GPU训练)

## 🛠️ 安装

### 1. 克隆代码库

```bash
git clone <repository-url>
cd final/code
```

### 2. 安装依赖

```bash
# 使用便捷脚本
python launcher.py setup

# 或手动安装
pip install -r requirements.txt
```

### 3. 数据准备

将翻译数据放置在 `data/` 目录下，格式为JSON：

```json
[
  {
    "english": "Hello, how are you?",
    "chinese": "你好，你好吗？"
  },
  ...
]
```

## 🚀 快速开始

### 1. 快速训练

```bash
# 使用便捷脚本训练BiLSTM模型
python launcher.py train bilstm

# 训练其他模型
python launcher.py train transformer
python launcher.py train lightweight
python launcher.py train pretrained
```

### 2. 自定义训练

```bash
# 完整训练命令
python main_trainer.py \
    --train_data_path data/translation2019zh_train.json \
    --model_type bilstm \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_accelerate \
    --mixed_precision fp16 \
    --log_dir logs/bilstm \
    --model_dir models/bilstm
```

### 3. 快速推理

```bash
# 使用便捷脚本
python launcher.py inference models/bilstm/best_model.pt "Hello world"

# 批量翻译
python launcher.py batch models/bilstm/best_model.pt input.txt output.txt
```

### 4. 模型评估

```bash
# 评估模型
python launcher.py evaluate models/bilstm/best_model.pt data/test_data.json
```

## 📊 模型架构

### 1. BiLSTM翻译器 (BiLSTMTranslator)

- **编码器**: 双向LSTM，捕获源语言的双向上下文
- **解码器**: 单向LSTM + 注意力机制
- **特点**: 参数量适中，训练稳定，适合中小规模数据

```python
# 模型配置示例
model_config = {
    'embedding_dim': 512,
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.1
}
```

### 2. Transformer翻译器 (TransformerTranslator)

- **架构**: 标准Transformer encoder-decoder结构
- **注意力**: 多头自注意力机制
- **特点**: 并行计算，长距离依赖建模能力强

```python
# 模型配置示例
model_config = {
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'dropout': 0.1
}
```

### 3. 轻量级翻译器 (LightweightTranslator)

- **编码器**: 预训练的多语言BERT
- **解码器**: 简化的Transformer解码器
- **特点**: 利用预训练知识，训练快速，效果好

### 4. 预训练翻译器 (PretrainedTranslator)

- **基础**: Helsinki-NLP/opus-mt-en-zh 或其他预训练模型
- **微调**: 支持全参数或部分参数微调
- **特点**: 零样本能力强，适合资源受限场景

## 🔧 高级功能

### 1. 数据加载优化

```python
# 智能数据加载器配置
from smart_dataset import create_smart_dataloaders

train_loader, val_loader, test_loader, vocab_info = create_smart_dataloaders(
    train_path='data/train.json',
    batch_size=32,
    tokenizer_type='transformers',
    cache_dir='cache',
    sample_ratio=0.1,  # 使用10%的数据进行快速实验
    num_workers=4,
    test_split=0.1,
    val_split=0.1
)
```

### 2. 多卡训练

```python
# 启用Accelerate
python main_trainer.py \
    --use_accelerate \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 2
```

### 3. 模型对比

```python
# 创建模型配置文件
model_configs = [
    {
        "name": "BiLSTM",
        "model_path": "models/bilstm/best_model.pt",
        "config_path": "configs/bilstm_config.json"
    },
    {
        "name": "Transformer",
        "model_path": "models/transformer/best_model.pt",
        "config_path": "configs/transformer_config.json"
    }
]

# 执行对比
python inference_new.py compare \
    --model_configs model_configs.json \
    --input_texts test_sentences.txt \
    --output_file comparison_results.json
```

### 4. 训练监控

系统会自动生成：
- **TensorBoard日志**: 实时训练曲线
- **训练图表**: loss、BLEU分数、学习率变化
- **检查点**: 自动保存最佳模型
- **日志文件**: 详细的训练过程记录

## 📈 性能优化

### 1. 内存优化

```python
# 大数据集处理
--sample_ratio 0.1          # 使用10%数据快速实验
--gradient_accumulation_steps 4  # 梯度累积，减少显存占用
--mixed_precision fp16       # 混合精度训练
```

### 2. 速度优化

```python
# 并行数据加载
--num_workers 4             # 4个进程并行加载数据
--cache_dir cache           # 启用缓存系统
```

### 3. 训练策略

```python
# 智能训练策略
--early_stopping 5          # 早停机制
--use_scheduler             # 学习率调度
--warmup_ratio 0.1          # 预热策略
```

## 🎯 使用场景

### 1. 研究实验

```bash
# 快速原型验证
python launcher.py train bilstm data/small_dataset.json 3

# 不同模型对比
python launcher.py train transformer
python launcher.py train lightweight
```

### 2. 生产部署

```bash
# 完整训练
python main_trainer.py \
    --train_data_path data/full_dataset.json \
    --model_type lightweight \
    --epochs 20 \
    --batch_size 64 \
    --use_accelerate \
    --mixed_precision fp16

# 性能评估
python launcher.py evaluate models/lightweight/best_model.pt data/test.json
```

### 3. 模型分析

```bash
# 多模型对比
python inference_new.py compare \
    --model_configs configs/all_models.json \
    --input_texts test_sentences.txt \
    --output_file comparison.json \
    --eval_data data/test.json
```

## 📁 项目结构

```
code/
├── main_trainer.py          # 主训练脚本
├── models.py               # 模型架构定义
├── smart_dataset.py        # 智能数据加载器
├── utils.py                # 工具函数
├── inference_new.py        # 推理和评估
├── launcher.py             # 便捷启动脚本
├── requirements.txt        # 依赖列表
├── README.md              # 说明文档
├── data/                  # 数据目录
│   ├── translation2019zh_train.json
│   └── translation2019zh_valid.json
├── models/                # 模型保存目录
├── logs/                  # 日志目录
├── cache/                 # 缓存目录
└── configs/               # 配置文件目录
```

## 🔍 常见问题

### Q1: 如何处理大数据集？

A: 使用采样和缓存机制：

```bash
python main_trainer.py \
    --sample_ratio 0.1 \
    --cache_dir cache \
    --num_workers 4
```

### Q2: 显存不足怎么办？

A: 使用混合精度和梯度累积：

```bash
python main_trainer.py \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4 \
    --batch_size 16
```

### Q3: 如何提升训练速度？

A: 启用多卡训练和优化数据加载：

```bash
python main_trainer.py \
    --use_accelerate \
    --mixed_precision fp16 \
    --num_workers 4
```

### Q4: 模型效果不好怎么办？

A: 尝试不同模型和调整超参数：

```bash
# 尝试预训练模型
python launcher.py train pretrained

# 调整学习率
python main_trainer.py --learning_rate 0.0001

# 增加训练轮次
python main_trainer.py --epochs 20
```

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目采用MIT许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: [your-email@example.com]
- 项目主页: [repository-url]

---

⭐ 如果这个项目对您有帮助，请给一个Star！
