# Findings and interpretation

## 1. R6 provides useful complementary ordering

R6's four-fold output reaches macro AUC **0.936579**, compared with **0.938369** for the complete baseline. Removing R6 reduces AUC to **0.923816**, a difference of **−0.014553** (conditional paired 95% interval **[−0.024026, −0.005397]**). In this fixed experiment, the remaining Raptor/Reader combination does not recover the discrimination of the complete ensemble.

This estimates the effect of bypassing a branch at inference with all other choices fixed. It is not a training ablation: models were not retrained without that component, and this is not an independent effect estimate for other populations.

## 2. Model quality and complementarity both matter

The SWA checkpoint has single-model AUC **0.932903** and average per-class Spearman correlation **0.895439** with the baseline 384 branch. At the already fixed extra-model weights it lifts the final ensemble to **0.940595**. The syouyatobita checkpoint has strong correlation (**0.971685**) but changes ensemble AUC by only **+0.000062**.

Low correlation alone is insufficient. royalacecat has correlation about **0.051632**, but under this particular inference harness its single AUC is **0.543925**, and adding it decreases ensemble AUC by **0.005224**. Checkpoint/preprocessing or label compatibility may explain its poor transfer; the experiment does not establish that the checkpoint is intrinsically poor.

The SWA contrast is **+0.002226** with a fixed-score interval **[−0.001006,+0.005772]**. Recomputing cohort ranks in the bootstrap gives **[−0.000751,+0.006916]**. Both cross zero. The positive-difference fractions (0.9175 and 0.942) should not be read as probabilities of future improvement. Gold58 screening and upstream selection are not corrected by these intervals.

## 3. The Reader changes ranks without changing AUC here

Reader removal changes **30 of 696 study-label rank cells**, by at most **1/58**. It changes **zero positive-negative pair orderings**. All 12 AUCs therefore remain equal; every fixed-score bootstrap AUC difference is exactly zero as well. The code independently checks that relationship.

This distinction matters: identical AUC does not mean identical predictions. Gold58 is small and the Reader weights are only 0.003–0.035. Its rank adjustments can be too small to change a discriminative ordering in this cohort. A larger untouched cohort and a runtime-matched comparison are needed before removing it from deployment.

## 4. Gains and errors are finding-dependent

Baseline AUCs are lowest for Synovitis (**0.836320**, 27 positives), Lateral OA (**0.847195**, 11 positives), and PF OA (**0.886744**, 21 positives). These are useful targets for future error review, but 58 studies cannot establish clinical error mechanisms or site-specific failures.

SWA improves six findings, ties four, and worsens two. Lateral Meniscus rises from **0.914286 to 0.927950**; Synovitis from **0.836320 to 0.847073**. Effusion falls from **0.980124 to 0.975155**, and Contusion from **0.960864 to 0.955466**. Per-class intervals, denominators and operating-point metrics are in `results/per_class.csv`.

At the fixed rank cutoff ≥0.5, baseline Recall is lowest for Synovitis (**21/27 = 0.777778**), PF OA (**17/21 = 0.809524**), and Effusion (**29/35 = 0.828571**). The ranks select roughly half the cohort; high Recall for a rare finding can coexist with poor specificity. Thresholds must be calibrated on separate data before clinical interpretation. No radiologist-reviewed image error categorization was performed.

## 5. More views did not show a clear, general gain

In the separate historical Apex view experiment, the reference AUC is **0.924601**. Increasing resolution to 320 gives **+0.000863**, flip TTA gives **+0.000496**, and k=8 gives **+0.000865**; all paired 95% intervals cross zero. Combining all V5 views gives **+0.000460**, again with an interval crossing zero.

This supports reporting sensitivity and uncertainty instead of assuming more preprocessing always helps. It does not establish that V13/V15 tolerates noise or intensity shifts, or that k=8 is globally best.

## 6. Development-set gains need a stronger validation design

The latest complete-label subset is reused development data. The historical local adapter split excludes Gold58 and has no train/dev study or normalized-report overlap, but that split belongs to a different local experiment. It cannot certify the public checkpoints used in V13/V15.

The official training table has no stable patient identifier. One normalized-report hash appears both inside and outside Gold58; repeated templates are a possible explanation. Patient-level independence remains **unverified**, and upstream checkpoint membership remains **unknown**.

As a separate leaderboard observation, the Gold58-tuned Gold Medal Opt recipe scored **0.957** while the V12/V13/V15 recipes reached **0.958**. These are rounded public leaderboard observations on another population, not a controlled causal comparison. They illustrate why development AUC and leaderboard outcomes should be recorded separately.

## 7. Runtime evidence does not prove end-to-end speedup

The four extra checkpoints cost approximately **2.01–5.71 seconds per study** in the shared 58-study T4 ×2 screen. SWA costs **2.42 seconds per study** there. Precision and caching differ across checkpoints. The SWA group has a conservative archived 1,300-study projection of **40,372 seconds**, above the original 7.5h projection gate. A later 3-example successful inference is a smoke test, not proof that the full hidden-test run fits the limit. GPU memory was not measured in this screen.

The account's latest V16 competition entry was still pending at evidence collection. This repository does not claim its public or private score is known.
