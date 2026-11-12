# S3a：经典深度适配基线（TempCNN 不适配、AdaBN、DANN），荷兰开发迁移（2026-11-06 写定，运行前）

- **性质**：论文必需的对照基线（PAPER_PLAN §6，经典锚点）。**不涉及本文方法**，不改冻结方法，符合"荷兰封板"。
- **代码**：直接调用冻结的 SAR/multiyear_20261017/deep_baseline_2023/db_run.py 中的 data、TempCNN、train、predict、adabn_predict，不做任何修改，**超参数原样**：25 epochs、batch 256、lr 1e-3、wd 1e-4、DANN λ 上限 0.3、种子 20261019 + k。
- **与 db_run 的唯一区别**：
  1. 迁移改为开发集的全部有序年份对（2017–2022、2024、2025；2023 不读），与 P9–P11 相同；
  2. 源年 TRAIN 去掉 xydrop 400 m 隔离区地块，与本线评估保持一致（db_run 原本没有 xydrop）。
- **输出**：每个迁移的 dl_raw、dl_adabn、dl_dann 的 BA，写入 s3a_deep_<grp>.csv。
- **只报告，不设判据**。比较对象：
  - P11 的 hgb_lw；
  - P10 的 hybridD_lw（高斯）；
  - TempCNN 本身。
- 按场景分解：测 A / 测 B；源年与目标年的偏移大小（以 hgb_raw 的跌幅分档）。
- 近三年方法（SFUDA RSE 2024、RAINCOAT ICML 2023）的实现另立 S3b 协议。
