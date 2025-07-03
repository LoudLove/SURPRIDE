## Ablation Study 实验内容说明

本目录下包含了一系列 ablation study（消融实验）相关的 Python 源码文件，用于系统性分析各组件及改进对模型性能的影响。文件命名中包含实验设计关键信息，方便追踪与对比。

### 主要文件及说明

- **最佳UHDIQA**  
  基线模型，实现了全部方法并达到最佳效果。

- **最佳UHDIQA_224 / use_deit224 / use_convnext224**  
  输入分辨率或骨干网络变体对比实验，`224` 代表输入图像大小为 224x224。

- **最佳UHDIQA_origin_conv_sr_deit / origin_deit_sr_conv / origin_deit_sr_deit**  
  对比不同的超分支路（SR branch）结构或主干网络设计，验证各自对性能的贡献。

- **最佳UHDIQA_use_deit384**  
  更高分辨率（384x384）下模型表现对比。

- **最佳UHDIQA_without_disloss / without_disloss_224**  
  移除判别损失（discriminator loss）或对抗损失，考察其作用。

- **最佳UHDIQA_without_SR_branch / without_SR_branch_224**  
  移除超分支（SR branch），评估该分支的重要性。

- **最佳UHDIQA_不同lambda_alpha间的实验**  
  以不同的 lambda 或 alpha 参数（通常为损失函数权重）进行消融实验，分析超参数敏感性。

---

### 实验目的

这些消融实验主要用于验证：

- 各模块（如SR分支、不同backbone等）对整体模型性能的影响
- 不同损失项对训练过程与结果的作用
- 不同输入分辨率或结构设计的优劣
- 关键超参数选择对模型的影响

通过系统性的消融对比，可以更有说服力地证明所提出方法中各组成部分的实际贡献。

---

> 备注：  
> 所有 `.py` 文件均为独立可运行的实验主程序，建议配合实验日志和指标对照查看。
