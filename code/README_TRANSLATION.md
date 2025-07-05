# 英中翻译模型训练项目

本项目基于test.ipynb实现了英文到中文的翻译模型训练和评价，包含T5和BiLSTM两个baseline模型的对比实验。

## 项目结构

```
code/
├── test.ipynb                          # 主要的训练notebook（已更新）
├── translation_trainer.py              # 完整的训练脚本
├── translation_evaluator.py            # 模型评估脚本
├── test_translation_models.py          # 功能测试脚本
├── smart_dataset.py                    # 智能数据集加载器
├── data/
│   ├── translation2019zh_train.json    # 训练数据
│   └── translation2019zh_valid.json    # 验证数据
├── t5_translation_model/               # T5模型保存目录
├── bilstm_translation_model.pth        # BiLSTM模型文件
├── model_comparison.png                # 训练对比图
└── translation_experiment_results.json # 实验结果
```

## 实验设计

### 1. 数据加载和预处理
- 使用Hugging Face `datasets`库加载JSON格式的翻译数据
- 与test.ipynb保持一致的数据处理方式
- 支持从训练集自动分割验证集

### 2. 模型对比

#### T5模型 (Baseline 1)
- **架构**: 基于预训练的T5-small模型
- **优势**: 
  - 训练简单，收敛快
  - 基于大规模预训练，效果好
  - 支持多种NLP任务
- **配置**:
  - 模型: `t5-small`
  - 最大长度: 512
  - 批次大小: 8
  - 学习率: 2e-5
  - 训练轮数: 3

#### BiLSTM模型 (Baseline 2)
- **架构**: 自定义的双向LSTM序列到序列模型
- **特点**:
  - 编码器: 双向LSTM
  - 解码器: 单向LSTM + 注意力机制
  - 从零开始训练
- **配置**:
  - 嵌入维度: 256
  - 隐藏维度: 512
  - 层数: 2
  - Dropout: 0.3
  - 词汇表大小: 3000 (英文) / 3000 (中文)

### 3. 评估指标
- **训练损失**: 监控模型收敛情况
- **训练时间**: 比较训练效率
- **BLEU分数**: 翻译质量评估（在evaluator中实现）
- **定性分析**: 人工检查翻译样例

## 使用方法

### 方法1: 使用Notebook（推荐）
```bash
# 打开notebook进行交互式训练
jupyter notebook test.ipynb
```

### 方法2: 使用脚本
```bash
# 1. 先测试环境
python test_translation_models.py

# 2. 运行完整训练
python translation_trainer.py

# 3. 评估模型
python translation_evaluator.py
```

### 方法3: 使用智能数据集
```python
from smart_dataset import create_huggingface_dataloaders

# 创建数据加载器
train_loader, valid_loader, tokenizer, data_collator = create_huggingface_dataloaders(
    train_data_path="data/translation2019zh_train.json",
    valid_data_path="data/translation2019zh_valid.json",
    tokenizer_name='t5-small',
    max_length=512,
    batch_size=16
)
```

## 环境要求

### 必需依赖
```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets>=2.0.0
pip install evaluate>=0.4.0
pip install matplotlib
pip install tqdm
pip install numpy
pip install scikit-learn
```

### 可选依赖
```bash
pip install sacrebleu  # BLEU评估
pip install jieba      # 中文分词
pip install spacy      # 英文分词
```

## 实验结果

### 训练效果对比
| 指标 | T5模型 | BiLSTM模型 |
|------|--------|------------|
| 训练难度 | 简单 | 复杂 |
| 收敛速度 | 快 | 慢 |
| 内存需求 | 高 | 中 |
| 可定制性 | 低 | 高 |
| 预期效果 | 好 | 中等 |

### 主要发现
1. **T5模型**适合快速原型开发和实际部署
2. **BiLSTM模型**适合学习和研究序列到序列模型
3. 两个模型都能产生基本的翻译结果
4. 实际应用需要更大数据集和更长训练时间

## 代码特点

### 1. 模块化设计
- 数据处理、模型定义、训练、评估分离
- 支持多种使用方式（notebook、脚本、API）

### 2. 完整的实验流程
- 环境设置 → 数据加载 → 模型训练 → 结果评估 → 模型保存

### 3. 错误处理和日志
- 详细的进度提示和错误处理
- 完整的实验记录和结果保存

### 4. 可扩展性
- 易于添加新的模型架构
- 支持不同的数据格式和评估指标

## 下一步改进

1. **数据增强**: 使用更大的翻译数据集
2. **模型优化**: 调整超参数，尝试不同架构
3. **评估完善**: 添加ROUGE、METEOR等评估指标
4. **部署优化**: 模型量化和推理加速
5. **多语言支持**: 扩展到其他语言对

## 许可证

本项目仅用于学习和研究目的。

## 贡献

欢迎提交Issue和Pull Request来改进项目！
