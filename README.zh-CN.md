# RSNA 膝关节异常检测：从榜单成绩到可复现实验

本仓库基于 **Xinyi Lu / andyyyyaa** 的实际比赛结果，完成逐类别评估、固定配方消融、配对不确定性分析和数据划分审计。

**已核实的最好公开榜成绩为 0.958。** 下文 Gold58 AUC 来自 58 例、12 类完整官方标签的开发集，不能与公开榜或最终私榜成绩混为一谈。V16 在本次取证时仍待评分，具体时间见 `results/competition_provenance.json`。

## 最值得说明的发现

| 比较 | Gold58 宏平均 AUC | 相对基线差值 | 差值的条件 95% 配对区间 |
|---|---:|---:|---:|
| 当前 V13/V15 配方 | 0.938369 | — | — |
| 去掉 R6 | 0.923816 | −0.014553 | [−0.024026, −0.005397] |
| 去掉 224 分支 | 0.939222 | +0.000853 | [−0.002200, +0.004294] |
| 去掉 Reader | 0.938369 | 0.000000 | [0, 0] |
| 加入固定权重 SWA | 0.940595 | +0.002226 | [−0.001006, +0.005772] |

1. **R6 是本次移除实验中贡献最大的分支。** 它单独的四折集成 AUC 为 0.936579，与其他分支有互补作用。
2. **SWA 的开发集增益存在，但证据不足以确认泛化提升。** 2,000 次配对 bootstrap 的区间跨 0；该候选已经在同一 Gold58 上经过四候选筛选，还存在选择偏差。
3. **Reader 不是“完全没有改变输出”。** 去掉它后 30 个 study-label 排名发生变化，但没有改变任何正负样本对的相对排序，因此 12 类 AUC 均不变。58 例的观察不足以证明它在更大测试集中没有价值。
4. **弱项不是所有病种均匀分布。** 基线 Synovitis AUC 为 0.836320，Lateral OA 为 0.847195，PF OA 为 0.886744。SWA 对 Lateral Meniscus 和 Synovitis 改善较大，同时降低了 Effusion 和 Contusion 的 AUC。
5. **Gold58 不是独立验证集。** 它被用于上游检查点选择及多轮开发。官方表没有稳定患者标识，无法证实当前公共模型的患者级训练/验证隔离。

![固定配方消融](figures/ablation_forest.png)

## 已交付的内容

- 18 个固定方案的消融结果及配对置信区间。
- 12 类 AUC、Recall、precision、specificity、FP/FN 数量和聚合错误共现分析。
- Gold58 与官方标签的一致性核查、历史本地微调划分核查，以及患者级审计的明确限制。
- 旧 Apex 模型的 11 组预处理敏感性分析，保留完整 21 配置汇总；与当前 V13/V15 分开解释。
- 最新 58 例 GPU 验证中各候选的实际耗时记录；没有用 3 例运行时间声称隐藏集加速。
- 可执行 Python 脚本、测试、已执行 Jupyter Notebook、PNG/SVG 图表、英文说明和方法文档。

Recall 的固定阈值是 **百分位排名 ≥0.5**，不是临床概率阈值。当前产物没有新做图像噪声/强度扰动，也没有补造 GPU 显存测量或独立患者验证结果。

## 如何复现

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/render_figures.py
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/research_audit.ipynb
```

公开的 Notebook 可以直接从汇总文件重新生成。重新计算 study 级指标需要你已有权限访问的原始预测缓存，命令见 [英文 README](README.md#reproduce) 与 [数据访问说明](docs/DATA_ACCESS.md)。公开仓库不包含影像、报告、Study ID、逐例预测、权重或凭据。

## 可以诚实使用的项目描述

> I conducted a reproducible retrospective ablation and per-class evaluation of an RSNA knee MRI ensemble, quantified conditional uncertainty with paired bootstrap resampling, and identified R6 complementarity, label-specific weaknesses, and limitations caused by repeated development-set reuse. The account achieved a verified public leaderboard score of 0.958; independent patient-level validation remains outstanding.

不要把这段工作描述为“独立验证证明提升”“自训了所有公共模型”或“已经达到临床诊断水平”。上游贡献与权重作者见 [ATTRIBUTION.md](docs/ATTRIBUTION.md)。
