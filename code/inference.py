import torch
import pickle
import jieba
from model import TransformerTranslator

class TranslationInference:
    """翻译推理类"""
    
    def __init__(self, model_path, vocab_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            self.vocab_en, self.vocab_zh = pickle.load(f)
        
        # 创建反向词汇表
        self.id_to_en = {v: k for k, v in self.vocab_en.items()}
        self.id_to_zh = {v: k for k, v in self.vocab_zh.items()}
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"English vocab size: {len(self.vocab_en)}")
        print(f"Chinese vocab size: {len(self.vocab_zh)}")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        model = TransformerTranslator(
            src_vocab_size=len(self.vocab_en),
            tgt_vocab_size=len(self.vocab_zh),
            d_model=512,
            n_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048,
            max_len=512,
            dropout=0.1
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def preprocess_text(self, text):
        """预处理英文文本"""
        # 清理文本
        text = text.strip().lower()
        
        # 分词
        tokens = text.split()
        
        # 转换为ID
        token_ids = [self.vocab_en.get(token, self.vocab_en['<unk>']) for token in tokens]
        
        return token_ids
    
    def postprocess_text(self, token_ids):
        """后处理中文文本"""
        # 转换为tokens
        tokens = [self.id_to_zh.get(id, '<unk>') for id in token_ids]
        
        # 移除特殊符号
        tokens = [token for token in tokens if token not in ['<sos>', '<eos>', '<pad>', '<unk>']]
        
        # 拼接成句子
        return ''.join(tokens)
    
    def translate(self, text, max_len=100, beam_size=1):
        """翻译文本"""
        # 预处理
        src_ids = self.preprocess_text(text)
        src_tensor = torch.tensor([src_ids]).to(self.device)
        
        # 生成翻译
        with torch.no_grad():
            if beam_size == 1:
                # 贪婪搜索
                output = self.model.generate(
                    src_tensor, 
                    max_len=max_len,
                    sos_idx=self.vocab_zh['<sos>'],
                    eos_idx=self.vocab_zh['<eos>']
                )
                output_ids = output[0].cpu().numpy()
            else:
                # 束搜索
                output_ids = self.beam_search(src_tensor, max_len, beam_size)
        
        # 后处理
        translation = self.postprocess_text(output_ids)
        
        return translation
    
    def beam_search(self, src, max_len=100, beam_size=5):
        """束搜索生成"""
        # 编码源序列
        src_mask = self.model.create_padding_mask(src)
        encoder_output = self.model.encode(src, src_mask)
        
        # 初始化候选序列
        batch_size = src.size(0)
        sos_idx = self.vocab_zh['<sos>']
        eos_idx = self.vocab_zh['<eos>']
        
        # 候选序列：[batch_size, beam_size, seq_len]
        candidates = torch.zeros(batch_size, beam_size, 1, dtype=torch.long, device=src.device)
        candidates[:, :, 0] = sos_idx
        
        # 候选序列分数
        candidate_scores = torch.zeros(batch_size, beam_size, device=src.device)
        candidate_scores[:, 1:] = -float('inf')  # 只有第一个候选有效
        
        for step in range(max_len):
            # 获取当前所有候选的下一个token概率
            all_candidates = []
            all_scores = []
            
            for beam_idx in range(beam_size):
                current_seq = candidates[:, beam_idx, :]
                
                # 如果序列已结束，跳过
                if current_seq[0, -1] == eos_idx:
                    all_candidates.append(current_seq)
                    all_scores.append(candidate_scores[:, beam_idx])
                    continue
                
                # 生成下一个token的概率
                tgt_mask = self.model.create_padding_mask(current_seq)
                look_ahead_mask = self.model.create_look_ahead_mask(current_seq.size(1)).to(current_seq.device)
                tgt_mask = tgt_mask & look_ahead_mask
                
                decoder_output = self.model.decode(current_seq, encoder_output, src_mask, tgt_mask)
                next_token_logits = self.model.output_projection(decoder_output[:, -1, :])
                next_token_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # 计算新的候选分数
                new_scores = candidate_scores[:, beam_idx].unsqueeze(1) + next_token_probs
                
                # 获取top-k候选
                top_scores, top_indices = torch.topk(new_scores, beam_size, dim=-1)
                
                for k in range(beam_size):
                    new_seq = torch.cat([current_seq, top_indices[:, k].unsqueeze(1)], dim=1)
                    all_candidates.append(new_seq)
                    all_scores.append(top_scores[:, k])
            
            # 选择最佳的beam_size个候选
            all_scores = torch.stack(all_scores, dim=1)
            top_scores, top_indices = torch.topk(all_scores, beam_size, dim=-1)
            
            new_candidates = torch.zeros_like(candidates)
            for i, idx in enumerate(top_indices[0]):
                new_candidates[:, i, :] = all_candidates[idx]
            
            candidates = new_candidates
            candidate_scores = top_scores
            
            # 如果所有候选都结束了，停止
            if torch.all(candidates[:, :, -1] == eos_idx):
                break
        
        # 返回最佳候选
        best_candidate = candidates[0, 0, :].cpu().numpy()
        return best_candidate
    
    def translate_file(self, input_file, output_file):
        """翻译文件"""
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        translations = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                translation = self.translate(line)
                translations.append(translation)
                print(f"[{i+1}/{len(lines)}] {line} -> {translation}")
            else:
                translations.append("")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        
        print(f"Translations saved to {output_file}")

def main():
    """主函数"""
    # 创建推理器
    translator = TranslationInference(
        model_path="checkpoints/best_model.pth",
        vocab_path="vocab.pkl",
        device='cuda'
    )
    
    # 交互式翻译
    print("=== 英译中翻译系统 ===")
    print("输入 'quit' 退出")
    print("输入 'file' 进行文件翻译")
    
    while True:
        user_input = input("\n请输入英文句子: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'file':
            input_file = input("请输入输入文件路径: ")
            output_file = input("请输入输出文件路径: ")
            translator.translate_file(input_file, output_file)
        else:
            translation = translator.translate(user_input)
            print(f"翻译结果: {translation}")

if __name__ == "__main__":
    main()
