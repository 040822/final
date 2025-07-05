"""
训练工具函数
包含日志设置、检查点保存/加载、评估指标、可视化等功能
"""
import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sacrebleu import corpus_bleu, sentence_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("⚠️  SacreBLEU不可用，将使用简化的BLEU计算")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def setup_logging(log_dir):
    """设置日志系统"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已设置，日志文件: {log_file}")
    
    return logger

def save_checkpoint(model, optimizer, scheduler, epoch, loss, bleu_score, 
                   config, save_path, is_best=False):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'bleu_score': bleu_score,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存检查点
    torch.save(checkpoint, save_path)
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = save_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f"✓ 最佳模型已保存: {best_path}")
    
    print(f"✓ 检查点已保存: {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载训练检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    bleu_score = checkpoint.get('bleu_score', 0.0)
    
    print(f"✓ 检查点已加载: {checkpoint_path}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}, BLEU: {bleu_score:.4f}")
    
    return epoch, loss, bleu_score

def calculate_bleu_score(predictions, references, tokenizer=None):
    """计算BLEU分数"""
    if not predictions or not references:
        return 0.0
    
    try:
        if SACREBLEU_AVAILABLE:
            # 使用SacreBLEU
            if isinstance(predictions[0], list) and isinstance(references[0], list):
                # Token级别
                pred_strings = []
                ref_strings = []
                
                for pred, ref in zip(predictions, references):
                    if tokenizer:
                        pred_str = tokenizer.decode(pred, skip_special_tokens=True)
                        ref_str = tokenizer.decode(ref[0] if isinstance(ref, list) else ref, skip_special_tokens=True)
                    else:
                        pred_str = ' '.join(map(str, pred))
                        ref_str = ' '.join(map(str, ref[0] if isinstance(ref, list) else ref))
                    
                    pred_strings.append(pred_str)
                    ref_strings.append(ref_str)
                
                # 计算corpus BLEU
                bleu = corpus_bleu(pred_strings, [ref_strings])
                return bleu.score / 100.0  # 转换为0-1范围
            else:
                # 字符串级别
                bleu = corpus_bleu(predictions, [references])
                return bleu.score / 100.0
        else:
            # 简化的BLEU计算
            return simple_bleu_score(predictions, references)
    
    except Exception as e:
        print(f"BLEU计算失败: {e}")
        return 0.0

def simple_bleu_score(predictions, references, max_n=4):
    """简化的BLEU分数计算"""
    if not predictions or not references:
        return 0.0
    
    scores = []
    
    for pred, ref in zip(predictions, references):
        if isinstance(ref, list):
            ref = ref[0]  # 取第一个参考翻译
        
        # 转换为字符串列表
        if isinstance(pred, list):
            pred = [str(x) for x in pred]
        if isinstance(ref, list):
            ref = [str(x) for x in ref]
        
        # 计算n-gram精度
        ngram_scores = []
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            if len(pred_ngrams) == 0:
                ngram_scores.append(0.0)
                continue
            
            # 计算匹配的n-gram数量
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(pred_ngrams.values())
            ngram_scores.append(precision)
        
        # 计算几何平均
        if all(score > 0 for score in ngram_scores):
            geo_mean = np.exp(np.mean(np.log(ngram_scores)))
        else:
            geo_mean = 0.0
        
        # 简化的brevity penalty
        bp = min(1.0, len(pred) / len(ref)) if len(ref) > 0 else 0.0
        
        bleu = bp * geo_mean
        scores.append(bleu)
    
    return np.mean(scores)

def get_ngrams(tokens, n):
    """获取n-gram计数"""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams

def plot_training_curves(train_losses, val_losses, bleu_scores, save_path=None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # BLEU分数曲线
    axes[0, 1].plot(bleu_scores, label='BLEU Score', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('BLEU Score')
    axes[0, 1].set_title('BLEU Score Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss差值曲线
    if len(train_losses) == len(val_losses):
        loss_diff = [val - train for train, val in zip(train_losses, val_losses)]
        axes[1, 0].plot(loss_diff, label='Val - Train Loss', color='orange', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].set_title('Overfitting Monitor (Val - Train Loss)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 训练指标汇总
    axes[1, 1].axis('off')
    summary_text = f"""
    训练指标汇总:
    
    最佳BLEU分数: {max(bleu_scores) if bleu_scores else 0:.4f}
    最低训练Loss: {min(train_losses) if train_losses else 0:.4f}
    最低验证Loss: {min(val_losses) if val_losses else 0:.4f}
    
    总训练轮次: {len(train_losses)}
    最终BLEU: {bleu_scores[-1] if bleu_scores else 0:.4f}
    最终训练Loss: {train_losses[-1] if train_losses else 0:.4f}
    最终验证Loss: {val_losses[-1] if val_losses else 0:.4f}
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存: {save_path}")
    
    plt.show()
    return fig

def evaluate_model(model, dataloader, criterion, device, tokenizer=None, max_batches=None):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    predictions = []
    references = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # 获取数据
            if len(batch) == 4:
                src_ids, src_mask, tgt_ids, tgt_mask = batch
                src_ids, src_mask = src_ids.to(device), src_mask.to(device)
                tgt_ids, tgt_mask = tgt_ids.to(device), tgt_mask.to(device)
            else:
                src_ids, tgt_ids = batch
                src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
                src_mask = tgt_mask = None
            
            # 前向传播
            if hasattr(model, 'generate') and hasattr(model, 'tokenizer'):
                # 预训练模型
                if src_mask is not None:
                    outputs = model(src_ids, src_mask, tgt_ids[:, :-1], tgt_mask[:, :-1] if tgt_mask is not None else None)
                else:
                    outputs = model(src_ids, tgt_ids[:, :-1])
                
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_ids[:, 1:].reshape(-1))
                
                # 生成预测
                if src_mask is not None:
                    generated = model.generate(src_ids, src_mask, max_length=tgt_ids.size(1))
                else:
                    generated = model.generate(src_ids, max_length=tgt_ids.size(1))
                
                predictions.extend(generated.cpu().numpy())
                references.extend(tgt_ids.cpu().numpy())
                
            else:
                # 其他模型
                if src_mask is not None:
                    outputs = model(src_ids, src_mask, tgt_ids[:, :-1], tgt_mask[:, :-1] if tgt_mask is not None else None)
                else:
                    outputs = model(src_ids, tgt_ids[:, :-1])
                
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_ids[:, 1:].reshape(-1))
                
                # 贪心解码预测
                pred_tokens = outputs.argmax(dim=-1)
                predictions.extend(pred_tokens.cpu().numpy())
                references.extend(tgt_ids[:, 1:].cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # 计算BLEU分数
    bleu_score = calculate_bleu_score(predictions, references, tokenizer)
    
    return {
        'loss': avg_loss,
        'bleu': bleu_score,
        'predictions': predictions[:10],  # 保存前10个预测用于检查
        'references': references[:10]
    }

def save_config(config, save_path):
    """保存配置文件"""
    config_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            config_dict[key] = value
        else:
            config_dict[key] = str(value)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 配置已保存: {save_path}")

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    print(f"✓ 配置已加载: {config_path}")
    return config_dict

def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def get_model_size(model):
    """获取模型大小（MB）"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def create_summary_report(config, results, save_path):
    """创建训练总结报告"""
    report = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'model_type': config.model_type,
            'dataset': config.train_data_path,
            'total_epochs': config.epochs
        },
        'model_config': {
            'max_len': config.max_len,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'dropout': config.dropout
        },
        'training_results': results,
        'best_metrics': {
            'best_bleu': max(results.get('bleu_scores', [0])),
            'best_val_loss': min(results.get('val_losses', [float('inf')])),
            'final_bleu': results.get('bleu_scores', [0])[-1] if results.get('bleu_scores') else 0,
            'final_val_loss': results.get('val_losses', [float('inf')])[-1] if results.get('val_losses') else float('inf')
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 训练报告已保存: {save_path}")
    return report

if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试BLEU计算
    predictions = [["hello", "world"], ["good", "morning"]]
    references = [[["hello", "world"]], [["good", "morning"]]]
    
    bleu = simple_bleu_score(predictions, references)
    print(f"BLEU分数: {bleu:.4f}")
    
    # 测试参数统计
    import torch.nn as nn
    model = nn.Linear(100, 50)
    params = count_parameters(model)
    print(f"模型参数: {params}")
    
    print("工具函数测试完成!")
