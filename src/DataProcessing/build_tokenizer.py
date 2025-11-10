import os
import sentencepiece as spm

def train_spm(input_txt: str, model_prefix: str, vocab_size: int = 10000, model_type: str = "bpe"):
    """
    训练一个 SentencePiece 子词分词器并保存模型文件（.model / .vocab）。
    参数：
    input_txt: 用于训练分词器的语料路径（文本文件，建议每行一个句子）
    model_prefix: 输出文件前缀（例如 'processed_data/spm_en'），会生成 'spm_en.model' 等
    vocab_size: 词表大小（子词数量），需与 Transformer 的 src/tgt_vocab_size 保持一致
    model_type: 分词器类型，常用 'bpe'（Byte-Pair Encoding）
    
    说明：
    pad_id=0, unk_id=1, bos_id=2, eos_id=3 固定了特殊符号的 ID，需与数据集/模型中保持一致
    character_coverage=1.0 用于覆盖语料的字符集比例
    """
    spm.SentencePieceTrainer.train(
        input=input_txt,                 # 训练语料文件路径
        model_prefix=model_prefix,       # 输出模型文件前缀
        vocab_size=vocab_size,           # 词表大小（子词数量）
        model_type=model_type,           # 分词器类型（BPE）
        character_coverage=1.0,          # 字符覆盖率（EN/DE 用 1.0）
        pad_id=0, unk_id=1, bos_id=2, eos_id=3  # 特殊符号的固定 ID，需与下游一致
    )
    print(f"已训练分词器: {model_prefix}.model")

def main():
    """
    主流程：
    指向处理后的平行语料目录（processed_data）
    分别用英文/德文训练两个 SentencePiece 分词器（词表默认大小 10000）
    输出 'spm_en.model' 与 'spm_de.model'，供后续数据编码与模型初始化使用

    注意：
    训练得到的实际词表大小以分词器接口返回为准（后续用 get_piece_size() 读取）
    """
    base_dir = os.path.join(os.path.dirname(__file__), "processed_data")  # 已清洗数据所在目录
    print(f"已指向处理后的平行语料目录: {base_dir}")
    train_en = os.path.join(base_dir, "train.en")  # 英文训练语料
    train_de = os.path.join(base_dir, "train.de")  # 德文训练语料

    # 英文/德文各自训练一个分词器
    train_spm(train_en, os.path.join(base_dir, "spm_en"), vocab_size=10000)
    train_spm(train_de, os.path.join(base_dir, "spm_de"), vocab_size=10000)

if __name__ == "__main__":
    main()