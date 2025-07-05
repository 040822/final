"""
便捷启动脚本
提供预设的配置和常用命令，方便快速开始训练和推理
"""
import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("✓ 命令执行成功")
            if result.stdout:
                print(result.stdout)
        else:
            print("✗ 命令执行失败")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ 命令执行异常: {e}")
        return False

def create_model_config(model_type, output_path):
    """创建模型配置文件"""
    configs = {
        'bilstm': {
            'model_type': 'bilstm',
            'tokenizer_type': 'transformers',
            'tokenizer_name': 'bert-base-multilingual-cased',
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10,
            'embedding_dim': 512,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.1,
            'max_len': 128,
            'use_accelerate': True,
            'mixed_precision': 'fp16'
        },
        'transformer': {
            'model_type': 'transformer',
            'tokenizer_type': 'transformers',
            'tokenizer_name': 'bert-base-multilingual-cased',
            'batch_size': 16,
            'learning_rate': 0.0001,
            'epochs': 15,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'max_len': 128,
            'use_accelerate': True,
            'mixed_precision': 'fp16'
        },
        'lightweight': {
            'model_type': 'lightweight',
            'tokenizer_type': 'transformers',
            'tokenizer_name': 'bert-base-multilingual-cased',
            'batch_size': 64,
            'learning_rate': 0.0001,
            'epochs': 8,
            'hidden_dim': 512,
            'dropout': 0.1,
            'max_len': 128,
            'use_accelerate': True,
            'mixed_precision': 'fp16'
        },
        'pretrained': {
            'model_type': 'pretrained',
            'tokenizer_type': 'transformers',
            'pretrained_model_name': 'Helsinki-NLP/opus-mt-en-zh',
            'batch_size': 32,
            'learning_rate': 0.00001,
            'epochs': 5,
            'dropout': 0.1,
            'max_len': 128,
            'use_accelerate': True,
            'mixed_precision': 'fp16'
        }
    }
    
    if model_type not in configs:
        print(f"❌ 不支持的模型类型: {model_type}")
        return False
    
    config = configs[model_type]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 配置文件已创建: {output_path}")
    return True

def setup_environment():
    """设置环境"""
    print("设置Python环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 安装依赖
    print("安装依赖包...")
    return run_command(
        "pip install -r requirements.txt",
        "安装requirements.txt中的依赖"
    )

def quick_train(model_type='bilstm', data_path='data/translation2019zh_train.json', epochs=5):
    """快速训练"""
    print(f"开始快速训练 - 模型类型: {model_type}")
    
    # 创建配置文件
    config_path = f"configs/{model_type}_config.json"
    os.makedirs("configs", exist_ok=True)
    
    if not create_model_config(model_type, config_path):
        return False
    
    # 构建训练命令
    cmd = f"""python main_trainer.py \
        --train_data_path "{data_path}" \
        --model_type {model_type} \
        --epochs {epochs} \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --log_dir "logs/{model_type}" \
        --model_dir "models/{model_type}" \
        --cache_dir "cache" \
        --sample_ratio 0.1 \
        --use_accelerate \
        --mixed_precision fp16 \
        --early_stopping 3"""
    
    return run_command(cmd, f"训练{model_type}模型")

def quick_inference(model_path, text="Hello, how are you?"):
    """快速推理"""
    print(f"使用模型进行快速推理")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    cmd = f'python inference_new.py translate --model_path "{model_path}" --text "{text}"'
    return run_command(cmd, "快速翻译测试")

def batch_translate(model_path, input_file, output_file):
    """批量翻译"""
    print(f"执行批量翻译")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    cmd = f'python inference_new.py batch --model_path "{model_path}" --input_file "{input_file}" --output_file "{output_file}"'
    return run_command(cmd, "批量翻译")

def evaluate_model(model_path, test_data_path, output_file=None):
    """评估模型"""
    print(f"评估模型性能")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not os.path.exists(test_data_path):
        print(f"❌ 测试数据不存在: {test_data_path}")
        return False
    
    cmd = f'python inference_new.py evaluate --model_path "{model_path}" --data_path "{test_data_path}"'
    if output_file:
        cmd += f' --output_file "{output_file}"'
    
    return run_command(cmd, "模型评估")

def main():
    """主函数"""
    print("🚀 英译中翻译系统启动器")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python launcher.py setup                    # 设置环境")
        print("  python launcher.py train [model_type]       # 快速训练")
        print("  python launcher.py inference [model_path]   # 快速推理")
        print("  python launcher.py batch [model_path] [input_file] [output_file]  # 批量翻译")
        print("  python launcher.py evaluate [model_path] [test_data]  # 评估模型")
        print("\n支持的模型类型: bilstm, transformer, lightweight, pretrained")
        return
    
    command = sys.argv[1]
    
    if command == "setup":
        setup_environment()
    
    elif command == "train":
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'bilstm'
        data_path = sys.argv[3] if len(sys.argv) > 3 else 'data/translation2019zh_train.json'
        epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        quick_train(model_type, data_path, epochs)
    
    elif command == "inference":
        if len(sys.argv) < 3:
            print("❌ 请提供模型路径")
            return
        model_path = sys.argv[2]
        text = sys.argv[3] if len(sys.argv) > 3 else "Hello, how are you?"
        quick_inference(model_path, text)
    
    elif command == "batch":
        if len(sys.argv) < 5:
            print("❌ 请提供模型路径、输入文件和输出文件")
            return
        model_path = sys.argv[2]
        input_file = sys.argv[3]
        output_file = sys.argv[4]
        batch_translate(model_path, input_file, output_file)
    
    elif command == "evaluate":
        if len(sys.argv) < 4:
            print("❌ 请提供模型路径和测试数据路径")
            return
        model_path = sys.argv[2]
        test_data_path = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        evaluate_model(model_path, test_data_path, output_file)
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == "__main__":
    main()
