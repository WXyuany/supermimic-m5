# 来源与许可

代码基于 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，固定来源 commit `d9be385b3eeb13a638fb33fb200d3bfd9be9839f`。原 MIT 许可证及 Copyright (c) 2026 Yuxi Liu 保留于 [LICENSE](LICENSE)。本仓库新增 Apple M5/macOS 适配、便捷脚本、M5 实验记录及展示；并非原论文作者官方实现。上游原始 README 可在上述固定 commit 查看，其中 NVIDIA 数字属于上游，不能当作本仓库 M5 实验。

仿真依赖 [mjlab](https://github.com/mujocolab/mjlab) commit `8bb2bc9e187797689ec82c78c35019afd2fd0376`（Apache-2.0），以及 MuJoCo、MuJoCo Warp、NVIDIA Warp、PyTorch；各依赖及机器人资产保留各自许可。本仓库不复制依赖中的机器人资产。

数据下载自 [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)，revision `ce1572906efe6157840e8474d5a0d7aa87481e74`。数据集页面指出原始 [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 数据使用 **CC BY-NC-ND 4.0**。本仓库不重新分发原始 CSV 或转换后的 NPZ；脚本从该来源下载。MIT 仅适用于本仓库代码，不会替代数据、视频中参考动作或模型使用所涉及的数据许可。提供的模型和演示用于研究复核；其他用途请核对原数据和相关资产的条款。

公开模型来自本次 M5 实验。实验报告、图表和视频明确区分上游工作、Mac 实测和未验证推断。
