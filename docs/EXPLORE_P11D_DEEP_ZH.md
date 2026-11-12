# P11d：C2 的深度网络部分，冻结方法的重合成 + TempCNN / DANN（2026-11-06 写定，运行前）

- **起因**：S3a 显示 DANN（TempCNN）整体略强于冻结方法（测 B −0.009，测 A −0.006）。冻结方法只在偏移最严重的三分之一迁移上占优。PAPER_PLAN 的 E4 本来就计划在深度网络上检验 C2"与分类器无关"。问题是：用冻结方法估出的偏移改写源年数据，能不能与 DANN 叠加？
- **性质**：评估冻结方法的输出（Δ），**不改方法**，不改 TempCNN/DANN 的任何超参数，符合荷兰封板。
- **做法**：
  - 每个迁移、每一折，用冻结模块（frozen/hybridD_lw.py，档案取另一站点）估计 Δ_lw；对照用 em_cov01 的 Δ_em。
  - 把 Δ 加到源年 TRAIN 序列上：VV、VH 按日期平移，VH−VV 重新计算，COH 按槽平移（序列中 COH 重复两次，Δ 也重复两次）。
  - 标准化统计量用平移后的源年 TRAIN 计算。
  - 训练 TempCNN 和 DANN，代码与 S3a 相同（db_run.train，同一种子）。
- **变体**：tcn_em、tcn_lw、dann_lw。参照值取 S3a 的 dl_raw、dl_dann（同一代码和种子）。
- **迁移**：开发集全部有序年份对，2023 不读，源年 TRAIN 使用 xydrop。
- **判据**（两个测试方向分别计算配对差）：
  - **D1 主**：dann_lw 对比 dl_dann，两个方向中位数都 > 0，且都至少 60% 为正，即与 DANN 互补。
  - **D2**：tcn_lw 对比 tcn_em，两个方向中位数都 ≥ +0.01。
  - 只报告：tcn_lw 对比 dl_raw；dann_lw 对比 gauss_lw；按偏移严重程度分档的结果。
- **输出**：p11d_deep_<grp>.csv。
