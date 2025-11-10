import os
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

WHITESPACE_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    """
    对输入字符串做最小清洗：
    - 去除首尾空白
    - 将连续空白字符（空格、Tab、换行等）规范化为单个空格
    返回：清洗后的字符串
    """
    s = s.strip()  # 去掉首尾空白
    s = WHITESPACE_RE.sub(" ", s)  # 将任意长度的空白序列压缩为一个空格
    return s

def load_train_pairs(train_en_path: str, train_de_path: str) -> List[Tuple[str, str]]:
    """
    从训练数据 train.tags.en-de.en / train.tags.en-de.de 读取英德平行句对。
    处理逻辑：
    - 过滤掉以 '<' 开头的标签行（例如 <doc>、<seg id="..."> 等）
    - 只保留非空纯文本行，并做基本清洗（空白规范化）
    - 英文与德文文件在清洗后按行一一对齐
    返回：[(英文句子, 德文句子), ...]
    """

    def read_clean_lines(fp: str) -> List[str]:
        """
        读取给定文件，跳过标签行并清洗文本，返回纯文本行列表。
        参数：
        - fp: 文件路径
        返回：清洗后的文本行列表
        """
        lines = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()  # 去掉首尾空白
                # 跳过标签行（IWSLT train.tags.* 中的元数据通常以 '<' 开头）
                if not line or line.startswith("<"):
                    continue
                # 基本清洗（空白规范化）
                lines.append(clean_text(line))
        return lines

    en_lines = read_clean_lines(train_en_path)  # 英文训练语料（已清洗）
    de_lines = read_clean_lines(train_de_path)  # 德文训练语料（已清洗）
    # 训练集必须保证英文与德文行数一一对应
    assert len(en_lines) == len(de_lines), f"训练集英德行数不匹配: {len(en_lines)} vs {len(de_lines)}"
    return list(zip(en_lines, de_lines))  # 拼成平行句对

def parse_xml_pairs(en_xml_path: str, de_xml_path: str) -> List[Tuple[str, str]]:
    """
    解析 dev/tst 的 XML 文件，按 <seg id="..."> 对齐生成英德平行句对。
    处理逻辑：
    - 分别解析英/德 XML，提取所有 seg 的 (id -> 文本) 映射
    - 找到英德共同的 id 集合，按 id 排序对齐
    - 返回 [(英文句子, 德文句子), ...]
    """

    def segs_by_id(fp: str) -> dict:
        """
        从 XML 文件中抽取所有 <seg id="..."> 的文本，形成 {id: 文本} 映射。
        参数：
        - fp: XML 文件路径
        返回：以整数 id 为键、清洗后文本为值的字典
        """
        tree = ET.parse(fp)           # 解析 XML
        root = tree.getroot()         # 获取根节点
        d = {}
        for seg in root.iter("seg"):  # 遍历所有 <seg> 节点
            sid = seg.attrib.get("id")
            if sid is None:
                continue
            text = seg.text or ""     # 取文本（空则用空字符串）
            d[int(sid)] = clean_text(text)  # id 转为整数并清洗文本
        return d

    en_map = segs_by_id(en_xml_path)  # 英文：id -> 文本
    de_map = segs_by_id(de_xml_path)  # 德文：id -> 文本
    # 只对齐两侧都存在的 id，并按 id 升序排序，保证稳定性
    common_ids = sorted(set(en_map.keys()) & set(de_map.keys()))
    pairs = [(en_map[i], de_map[i]) for i in common_ids]
    return pairs

def save_pairs(pairs: List[Tuple[str, str]], out_prefix: str):
    """
    将平行句对保存为两个按行对齐的文本文件：
    - {out_prefix}.en 保存英文，每行一个句子
    - {out_prefix}.de 保存德文，每行一个句子
    这样下游可以按行直接读取对应的英德样本。
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)  # 确保输出目录存在
    en_fp = f"{out_prefix}.en"
    de_fp = f"{out_prefix}.de"
    with open(en_fp, "w", encoding="utf-8") as fe, open(de_fp, "w", encoding="utf-8") as fd:
        for en, de in pairs:
            fe.write(en + "\n")
            fd.write(de + "\n")
    print(f"Saved: {en_fp} ({len(pairs)} lines), {de_fp} ({len(pairs)} lines)")

def main():
    """
    主流程：
    - 定位原始 IWSLT2017 EN-DE 数据目录（en-de）
    - 解析并清洗训练集（train.tags.*），保存到 processed/train.{en,de}
    - 解析并清洗验证集（dev2010），保存到 processed/dev.{en,de}
    - 依次解析并清洗各测试集（tst2010–tst2015），保存到 processed/tstYYYY.{en,de}
    """
    # base_dir 指向原始数据所在的 en-de 目录
    base_dir = os.path.join(os.path.dirname(__file__), "en-de")
    # out_dir 是处理后的平行文本输出目录，位于当前脚本同级的 processed_data/
    out_dir = os.path.join(os.path.dirname(__file__), "processed_data")
    os.makedirs(out_dir, exist_ok=True)

    # 训练集：从 train.tags.en-de.* 读取并清洗，按行对齐保存
    train_en = os.path.join(base_dir, "train.tags.en-de.en")
    train_de = os.path.join(base_dir, "train.tags.en-de.de")
    train_pairs = load_train_pairs(train_en, train_de)
    save_pairs(train_pairs, os.path.join(out_dir, "train"))

    # 验证集：dev2010（XML 格式，按 seg id 对齐）
    dev_en = os.path.join(base_dir, "IWSLT17.TED.dev2010.en-de.en.xml")
    dev_de = os.path.join(base_dir, "IWSLT17.TED.dev2010.en-de.de.xml")
    dev_pairs = parse_xml_pairs(dev_en, dev_de)
    save_pairs(dev_pairs, os.path.join(out_dir, "dev"))

    # 测试集：遍历 tst2010–tst2015，存在即解析保存，否则打印缺失提示
    for year in ["tst2010", "tst2011", "tst2012", "tst2013", "tst2014", "tst2015"]:
        en_xml = os.path.join(base_dir, f"IWSLT17.TED.{year}.en-de.en.xml")
        de_xml = os.path.join(base_dir, f"IWSLT17.TED.{year}.en-de.de.xml")
        if os.path.exists(en_xml) and os.path.exists(de_xml):
            pairs = parse_xml_pairs(en_xml, de_xml)
            save_pairs(pairs, os.path.join(out_dir, year))
        else:
            print(f"Skip {year}: missing files")

if __name__ == "__main__":
    main()