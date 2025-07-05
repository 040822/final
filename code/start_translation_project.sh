#!/bin/bash
# 英中翻译模型训练快速启动脚本

echo "🚀 英中翻译模型训练项目快速启动"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装，请先安装Python"
    exit 1
fi

echo "✓ Python环境检查通过"

# 检查必需的包
echo "📦 检查依赖包..."

required_packages=("torch" "transformers" "datasets" "evaluate" "matplotlib" "tqdm" "numpy")

for package in "${required_packages[@]}"; do
    if python -c "import $package" &> /dev/null; then
        echo "✓ $package 已安装"
    else
        echo "❌ $package 未安装"
        echo "请运行: pip install $package"
        exit 1
    fi
done

# 检查数据文件
echo "📊 检查数据文件..."

if [ -f "data/translation2019zh_train.json" ]; then
    echo "✓ 训练数据文件存在"
else
    echo "❌ 训练数据文件不存在: data/translation2019zh_train.json"
    echo "请确保数据文件放在正确位置"
    exit 1
fi

if [ -f "data/translation2019zh_valid.json" ]; then
    echo "✓ 验证数据文件存在"
else
    echo "⚠️  验证数据文件不存在，将从训练集分割"
fi

# 显示选项菜单
echo ""
echo "🎯 请选择运行方式:"
echo "1. 运行功能测试 (推荐第一次使用)"
echo "2. 运行完整训练脚本"
echo "3. 运行模型评估"
echo "4. 打开Jupyter Notebook"
echo "5. 查看项目说明"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "🔍 运行功能测试..."
        python test_translation_models.py
        ;;
    2)
        echo "🚀 运行完整训练..."
        python translation_trainer.py
        ;;
    3)
        echo "📊 运行模型评估..."
        if [ -d "t5_translation_model" ] || [ -f "bilstm_translation_model.pth" ]; then
            python translation_evaluator.py
        else
            echo "❌ 未找到训练好的模型，请先运行训练"
        fi
        ;;
    4)
        echo "📓 启动Jupyter Notebook..."
        if command -v jupyter &> /dev/null; then
            jupyter notebook test.ipynb
        else
            echo "❌ Jupyter未安装，请运行: pip install jupyter"
        fi
        ;;
    5)
        echo "📖 查看项目说明..."
        if [ -f "README_TRANSLATION.md" ]; then
            cat README_TRANSLATION.md
        else
            echo "❌ 说明文件不存在"
        fi
        ;;
    *)
        echo "❌ 无效选择，请输入1-5之间的数字"
        exit 1
        ;;
esac

echo ""
echo "✅ 执行完成！"
echo "📁 生成的文件可能包括:"
echo "   - t5_translation_model/ (T5模型)"
echo "   - bilstm_translation_model.pth (BiLSTM模型)"
echo "   - model_comparison.png (对比图)"
echo "   - translation_experiment_results.json (结果)"
