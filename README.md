实现了Encoder-Decoder结构的Transformer模型，并在IWSLT2017 (EN↔DE)上进行训练
results文件夹下有训练曲线图和性能对比表
scripts文件夹下有.sh文件，通过bash命令执行可以训练模型
src文件夹有源代码，包括数据处理和模型代码

文件夹结构：

results/:
  
  训练损失曲线、实验结果

scripts/:
  
  用于训练的脚本文件

src/DataProcessing/:

  数据集源文件、处理后的数据集、数据集处理和分词器代码

src/Transformer/:

  模型各部分代码

src/train.py:

  训练代码

src/inspect_ckpt.py:

  读取pt文件代码
      

注：
报告中有一个错误，在表2：模型不同超参数的对比中，我对比了不同头的数量、以及输入维度大小的影响，但是对比实验中只分别修改了num_heads和d_model，没有修改其他超参数，报告中的表有错误，正确的表在results文件夹下。
消融实验我是做了不使用位置编码的影响，也可以在results文件夹下查看效果对比。
