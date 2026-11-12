# 新地区独立确认：预注册草案（2026-11-06，Claude；地区待用户选定，提交 HyP3 前必须先问用户）

> - 本草案在新地区数据**下载和查看之前**写定。地区确定后只填写 §1 的事实字段，判据和方法不变。
> - 改判据需要用户签字，并记录在 §9。

## 1. 地区与数据（待填）

| 项 | 内容 |
|---|---|
| 地区 / AOI（lon/lat 包围盒） | 待定，候选见 REGION_SURVEY_ZH.md |
| 标签来源、许可、URL | 待定 |
| 年份 | 2017–2025 中所有可用年份。若 2023 可用，照常使用：新地区没有暴露问题 |
| 相对轨道 / 槽日历 | 待定。槽日历按 FREEZE §5 默认用荷兰 T88 |
| 类别映射表（标签字段 → 7 类） | **在下载标签之前写定**，见 §3 |
| HyP3 作业数和积分 | 由 plan_region.py 估算，**用户批准后才提交** |

## 2. 冻结方法与档案

- 方法：frozen/hybridD_lw.py（SHA256 见 frozen/SHA256_FROZEN.tsv）。不改任何代码和超参数。
- 档案：frozen/archive_NL_all.npz（荷兰全部，1,008 个任务）。不使用新地区的任何年份重新估计。
- 特征处理与荷兰完全相同（build_features_my.py 的定义）：
  - 地块值 = 地块内纯 40 m 像元线性值的均值；
  - VV/VH 转为 dB，COH 保持线性；
  - 每个地块每层都必须有有限值；
  - RTC 20 m gamma0 power；INSAR_GAMMA 10x2，相位滤波 0.6。

## 3. 类别映射与地块筛选（先写原则，具体代码映射在下载前写入附表）

- 7 类：PermGrass、TempGrass、Maize、Potato、Beet、WinterCereal、SpringCereal。
- 只保留能无歧义映射的作物代码。混合类和不明确的代码（例如未注明冬春的"谷物"）排除。
- 地块筛选和空间折的算法与荷兰完全相同（SAR/multiyear_20261017/build_labels_my.py）：
  - 40 m 网格；地块向内缓冲 √2·20 m；像元被多个地块覆盖时去掉；
  - 至少 2 个纯像元的地块才合格（eligible_ge2）；
  - 按 5 km 块分为 6 折，块到折的分配用新地区固定的种子 20261106，对排序后的块列表随机打乱，按 i mod 6 分配；
  - 与测试折相距 400 m 以内的训练地块记为 EXCLUDED_400M。
- 新地区的网格自己定义，参照 grid_spec 的格式（EPSG 用当地 UTM，40 m）。
- 荷兰的参考量：A 站 2017 年合格地块 22,311 个，其中冬季谷物 200、春季谷物 186、甜菜 194（LABEL_QC_2017.json）。荷兰本身的小类就很小。
- 若某类在某一年少于 30 个地块，该类在这一年不建模。这是冻结代码的行为，不另作处理。

## 4. 迁移与比较

- **迁移**：新地区所有有序年份对（源年有标签，目标年无标签），每一对按 6 折评估。
- **主比较（高斯框架）**：
  - hybridD_lw（荷兰档案）；
  - em_cov01：最强的无先验基线，不需要档案；
  - hpj_cov01：全秩联合先验，荷兰档案；
  - 不适配；
  - oracle：上限。
- **C2（HGB 重合成，冻结 HGB 参数）**：hgb_lw、hgb_em、hgb_hpj、hgb_raw、hgb_oracle。
- **实际部署对照**：荷兰全部年份加新地区源年合并训练 HGB；新地区源年单独训练 HGB。
- **近三年方法**（S3 实现后在荷兰与新地区同时报告，参数按原文默认值）：SFUDA（RSE 2024）、RAINCOAT（ICML 2023），以及时间允许时的 TimeMatch、DANN。

## 5. 判据（新地区一次运行，按全部迁移的配对差计算）

| 编号 | 比较 | 通过条件 | 支撑 |
|---|---|---|---|
| N1 主 | hybridD_lw 对比 em_cov01 | 中位数 ≥ +0.01，且至少 2/3 迁移为正 | 档案先验跨地区有用（C1、C3） |
| N2 | hybridD_lw 对比 hpj_cov01 | 中位数 ≥ −0.003（不劣） | 物理基在跨地区时不比纯经验先验差 |
| N3 | hgb_lw 对比 hgb_em | 中位数 ≥ +0.01，且至少 2/3 为正 | C2 |
| N4（只报告） | hybridD_lw 对比 hpj_cov01，按迁移拆分 | — | 跨地区是否放大物理基的优势 |
| N5（只报告） | 新地区自身 3 年档案对比荷兰 8 年档案（构造方式同荷兰 k3） | — | "省数据"在新地区的表现 |

- 统计：报告中位数、为正比例，以及按年份块自助的 90% 区间（迁移之间共享年份，区间只作参考）。

## 6. 结果解释（事先写定）

- **N1 和 N3 都通过**：论文的确认证据成立，按 C 立意写作。
- **N1 通过、N3 不通过**：C2 只写荷兰的开发证据，确认只写 C1/C3。
- **N1 不通过**：如实报告"跨地区不迁移"，并分析原因：类别映射、轨道日历、气候、档案不匹配。**不改方法再试**；之后怎么走由用户决定。

## 7. 运行纪律

- 下载标签 → 写入映射附表（下载后不改）→ 计算地块数 → 用户批准 HyP3 → 处理 → 一次运行全部判据。
- 运行前记录全部代码和数据的 SHA256。
- 结果在 CONFIRM_NEWREGION_ZH.md 中一次性报告。

## 8. 待用户决定

1. 选哪个地区（见 REGION_SURVEY_ZH.md）。
2. HyP3 积分：每年每轨 18 个 RTC 和 9 个 InSAR 作业，约 405 积分（每个作业 15 积分，按荷兰计划的单价，需重新核对）；6 年约 2,430 积分。
3. 是否保留相干性。冻结方法需要 COH 维；如果不做 InSAR 作业，冻结方法不能原样使用。所以建议保留。

## 9. 变更记录

- 2026-11-06：草案初版（地区未定）。
- **2026-11-06 用户决定**（原话："选法国FR-A，映射按你推荐的增补，同意重跑RAINCOA"；组会记录 e:\组会.txt）：
  - **地区**：法国 FR-A Pays de Caux，lon 0.55–1.10，lat 49.55–49.90。
  - **标签**：IGN RPG parcelles_graphiques，用 WFS 按包围盒获取；许可 Licence Ouverte 2.0；年份 2017–2024。
  - **映射**：FR_MAPPING_FROZEN.csv（含签字增补 MIE、RGA；PRL 排除），SHA256 4ABE3414…8BDC。
  - **Sentinel-1**：相对轨道 59，升轨（与荷兰 T88 同为升轨）；槽日历用 T88。
  - **HyP3**：批准，但先冒烟再批量。先提交 1 年（2023，27 个作业，约 405 积分）验证整条管线：SLC 处理、相干性中点、几何配准到 40 m 网格、地块特征。确认无误后再提交其余 7 年。
  - **档案**：
    - 主确认用 archive_NL_all；
    - **事先写定的次要敏感性分析**：只用 T88 升轨档案（A88 + B88），回答"轨道几何对先验迁移的影响"。档案在确认运行前由冻结的 Archive.build 从 p9_arch_A88 和 p9_arch_B88 生成，并记录哈希。
  - **近三年基线**：用官方 RAINCOAT/SFUDA 代码做统一的小范围调参，作为后台任务，不阻塞主线。
  - **论文定位**："偏移越大、档案越短，物理先验越有价值，并且能与深度适配互补"。不主张全面最优。
- **2026-11-06 傍晚，用户签字增补 N6**（原话"可以"，回复"签 N6"方案；起草：支线 B 会话）：
  - 增加次要分支 N6/N6b/N6c，检验天气条件化联合先验 j2 能否跨地区迁移。全文：toy_B/full/frozen_j2/N6_ADDENDUM_ZH.md。
  - 冻结物：j2_wc.py 与 archive_j2_NL_all.npz（等价校验见 VERIFY_J2.json，12 个迁移 max|diff| = 0）。
  - 法国降水：Météo-France 76679001 SOMMESNIL（按事先写定的规则选出），只用日降水 RR。
  - **不影响 N1–N5 的判定。**
  - **运行要求**：与 N1–N5 在同一批迁移上一次运行；不能在看过 N1–N5 之后才决定是否跑 N6。

- **2026-11-06 傍晚：运行规格定稿**（写定于任何法国跨年迁移运行之前；已看过的法国结果只有 2023 年冒烟检查的年内 BA）：
  - **主脚本**：confirm_fr.py。一次计算 N1–N6 全部比较，按迁移写出 CONFIRM_FR_rows.csv，过程中不打印 BA。判据由 confirm_fr_summary.py 计算，统计口径同 §5，包括年份块自助 90% 区间。
  - **迁移**：法国 8 年（2017–2024）的全部有序年份对，共 56 个；6 个空间折；源年 TRAIN 剔除跨年 400 m 隔离区（fr_xyexcl.py，算法同荷兰 build_xyexcl.py）。
  - **变体**：
    - 高斯框架：g_raw、em_cov01、hpj_cov01（荷兰全部 p1_rows 的 LW 联合先验）、lw_all（主）、lw_T88（次要）、j2（N6）、g_oracle；
    - HGB 重合成：hgb_raw、hgb_em、hgb_hpj、hgb_lw、hgb_oracle；
    - N5：lw_local3 与 hpj_local3。本地档案由该迁移以外的 6 个法国年份中抽 3 个构成，rng 种子 20261106，按迁移顺序抽取，抽样结果写入 CONFIRM_FR_N5_subsets.json。
  - **N6c**：冻结方法 lw_all 的估计 Δ 投影到日期基，得到日期系数，与法国 ΔAPI（Météo-France RR，j2_wc.api_for_task）计算 Spearman。
  - **深度基线**（TempCNN、AdaBN、DANN，以及重合成 + DANN）在主运行之后，用同一批迁移另跑，只报告。
  - **SHA256 前 16 位**：
  - ..\frozen\archive_NL_all.npz：1CB9D565CA835449
  - ..\frozen\archive_NL_T88.npz：C68B3392A0A65D6D
  - ..\frozen\hybridD_lw.py：BF2880CE8930BC91
  - ..\toy_B\full\frozen_j2\archive_j2_NL_all.npz：CF0DCDA7338B30FD
  - ..\toy_B\full\frozen_j2\fr_rr_daily.csv：341D22215E8605C2
  - ..\toy_B\full\frozen_j2\j2_wc.py：CCF93902A4DEF6AA
  - confirm_fr.py：8B74BBE1774C1FAA
  - confirm_fr_summary.py：A08F217AD5BB32F7
  - FR_MAPPING_FROZEN.csv：4ABE3414D7D2D0C2
  - pipeline\fr_build_labels.py：09A3161194AD6E4E
  - pipeline\fr_download_align.py：FEFE1F2E4922D566
  - pipeline\fr_features.py：F85710EC9BB4EFAE
  - pipeline\fr_fetch_labels.py：CF9FDBC8A270FA66
  - pipeline\fr_xyexcl.py：7EE4529E1DA73DDD
  - D:\sar_data\fr_a\features_FR.npz：0824B73DC00A4613
  - D:\sar_data\fr_a\xyexcl_FR.npz：74F91E6B5B60B54B
  - fr_download_align.py（加低内存模式后）：71DAAFC0AB862E38（只改了调度方式，配准算法不变）

- **2026-11-06 22:40 运行中偏离（只涉及只报告的 N5，其余不变）**：
  - 第 7 个迁移（2017→2024）的法国本地 3 年档案没有任何完整任务（SpringCereal 太少，无法估计真实偏移），Archive.build 在 LW 上崩溃。前 6 个迁移已写盘，BA 未被查看。
  - **规则（续跑前写定）**：本地档案的完整任务少于 10 个时，该迁移的 lw_local3、hpj_local3 记为 NaN（"本地档案退化"）。N5 只在有效迁移上报告，并给出有效个数。
  - 已写盘的 6 个迁移按同一规则重新计算完整任务数，只改 N5 两列，其余列不动、不查看。
  - confirm_fr.py 只改了 N5 分支；主变体（N1–N4、N6）的代码路径不变。
  - 修改后 confirm_fr.py 的 SHA256 前 16 位：E7B854AEB42CEC53

- **2026-11-07 用户批准**（原话"都批准了"）：在主运行之后，于同一批 56 个迁移上运行法国深度基线（只报告）。
  - 脚本：fr_deep.py。冻结的 db_run.TempCNN/train/predict/adabn_predict，超参数原样（25 epochs、种子 20261019 + k）。
  - 变体：dl_raw、dl_adabn、dl_dann；用 em_cov01 的 Δ 重合成后训练的 tcn_em；用冻结方法（archive_NL_all）的 Δ 重合成后训练的 tcn_lw 和 dann_lw。
  - 重合成和序列格式与 P11d 相同。
  - 报告：与 lw_all（高斯）、hgb_lw 在相同迁移上比较；按偏移严重程度（dl_raw 三分位）分档，与荷兰 S3a、P11d 的口径相同。

- **2026-11-10 发布前审计补记**：
  - **问题**：confirm_fr_summary.py 现在的 SHA256 前 16 位是 EA40B51CD5857D12，与上面登记的 A08F217AD5BB32F7 不一致。文件修改时间为 2026-11-07 00:15 UTC，与 CONFIRM_FR_rows.csv 写完、CONFIRM_FR_SUMMARY.json 生成是同一分钟。时间上对应 N5"本地档案退化记为 NaN"的规则，但当时没有在本节记录，原版本也无法恢复。
  - **核对**：直接用 CONFIRM_FR_rows.csv 按 §5 的口径重算 N1、N2、N3、N6 的中位差和正值比例，与 CONFIRM_FR_SUMMARY.json 逐项一致：
    - N1 +0.0169 / 0.929；
    - N2 +0.0042 / 0.714；
    - N3 +0.0254 / 0.875；
    - N6 +0.0029 / 0.643；
    - N5 有效迁移数 17。
  - **结论**：判据结果不受这处修改影响。自助区间沿用 JSON 中的数值，未独立重算。
