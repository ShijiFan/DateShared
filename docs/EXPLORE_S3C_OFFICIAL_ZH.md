# S3c：近三年方法用官方代码重跑，并做统一的小范围调参（2026-11-06 写定，运行前）

- **授权**：用户 2026-11-06 原话"同意重跑RAINCOA"；组会记录 e:\组会.txt 第 5 项："引入官方 RAINCOAT / SFUDA 代码调参，但不阻塞主线"，作为后台异步任务。
- **起因**：S3b 的重写版不能用。RAINCOAT 在测 A 上只有 0.348；SFUDA 与源模型几乎相同。
- **性质**：只调基线，不涉及本文方法；荷兰封板不受影响。

## 1. RAINCOAT（官方代码）

- **代码**：E:\research\_ext\s3b_src\raincoat\Raincoat-main。直接调用以下官方实现：
  - algorithms/RAINCOAT.py 的 RAINCOAT 类（tf_encoder、tf_decoder、update、correct）；
  - models/loss.py 的 SinkhornDistance；
  - models/models.py 的 classifier。
- **只做两处最小改动**：
  1. tf_decoder 中写死的 `irfft(n=128)` 改为 `n=序列长度`（2S = 16 或 18）；
  2. `pytorch_metric_learning` 环境中没有安装，它只在 `__init__` 中创建一个未使用的 ContrastiveLoss，因此用空的占位模块代替，不改环境。
- **配置**（参照官方 HAR 配置，按序列长度缩放）：
  - input_channels 4，sequence_len 2S，kernel 5，stride 1，dropout 0.5；
  - mid_channels 64，final_out_channels 128，features_len 1；
  - fourier_modes = 2S//2，out_dim = 2·modes + 128；
  - batch 128，weight_decay 1e-4；
  - 每个阶段 epoch 数与官方 trainer 一致：对齐阶段 num_epochs 轮，纠正阶段 num_epochs 轮，num_epochs = 40。
- **调参网格**：lr ∈ {5e-4, 1e-3, 2e-3}，即官方三套 hparams 中 RAINCOAT 的取值。

## 2. SFUDA（官方算法，沿用 S3b 的实现与改动 1–4）

- **调参网格**：基础 lr ∈ {1e-3（原文）, 1e-2, 5e-2}。骨干仍为基础 lr 的 0.1 倍，与原文一致。

## 3. 调参与评估

- **调参集**：4 个固定的开发迁移，事先写定为
  - B88：2017→2018、2019→2021；
  - A88：2018→2020、2022→2024。
  - 按 6 折的平均 BA 选出每种方法的最佳 lr。这一步用到了目标年标签，相当于给基线开了"开发集调参"的优待，与本文方法在荷兰开发集上定超参数的地位对等。
- **正式评估**：迁移取 todo[0::2]（各组 28 个，共 84 个），与 P9/P10 的曲线集合相同。调参集中与之重合的迁移照常保留，并在表中注明。
- 数据接口、源模型、xydrop、标准化与 S3a、S3b 完全相同。
- **只报告**，与 S3a、P10、P11d 在相同迁移上合表。

## 4. 输出

- s3c_tune.csv：调参结果。
- s3c_official_<grp>.csv：正式结果。
