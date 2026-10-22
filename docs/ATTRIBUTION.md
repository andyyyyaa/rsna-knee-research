# Sources and attribution

## Competition and account evidence

- [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection): official task and macro AUC definition. Dataset contributors and challenge organizers are credited on its overview page.
- [Xinyi Lu / andyyyyaa](https://www.kaggle.com/andyyyyaa): account under which the evaluated experiments and submissions were run.
- [V13, version 360244791](https://www.kaggle.com/code/andyyyyaa/rsna-knee-v13-r6-only-reader-r7-removed?scriptVersionId=360244791): verified 0.958 public score recipe.
- [V15, version 360646466](https://www.kaggle.com/code/andyyyyaa/rsna-knee-v15-efficient-v13-inference?scriptVersionId=360646466): same model recipe with engineering changes; main validation baseline.
- [V16 Gold58 screen](https://www.kaggle.com/code/andyyyyaa/rsna-knee-v16-gold58-weight-screen): source of the actual main-experiment branch predictions and matched candidate costs.
- [Apex parameter validation, version 356976389](https://www.kaggle.com/code/andyyyyaa/rsna-apex-parameter-validation?scriptVersionId=356976389): historical preprocessing sensitivity appendix.

Some account notebooks are private. Links identify provenance, not a claim of public access.

## Pretrained checkpoints and upstream methods

- [nartaa publication SWA weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007): CoAtNet/Raptor branches.
- [rostigor R6 package](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406), originating from the heliosli pipeline: four-fold R6 branch. [heliosli foundation checkpoint](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) is an upstream input of the original pipeline.
- [goodpjw2008 ConvNeXt Reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader): three-fold 2.5D Reader. [R6 + Reader reference notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958).
- Extra checkpoint authors: **royalacecat**, **syouyatobita**, **kawaharataishi**, **tbongc**. The screen packages are [coat-extra](https://www.kaggle.com/datasets/manatoshoji03/rsna-knee-coat-extra) and [coat-extra2](https://www.kaggle.com/datasets/manatoshoji03/rsna-knee-coat-extra2). Exact checkpoint filenames and hashes are in `results/frozen_model_spec.json`.
- [tbongc original checkpoint card](https://www.kaggle.com/datasets/tbongc/rsna-knee-raptor-v10geo-v7w-f-1-s42-imagenet).
- Historical Apex source: [hitarthjain0 Apex stack](https://www.kaggle.com/code/hitarthjain0/rsna-knee-apex-grandmaster-stack-v5).
- The class-specific Raptor/R6 recipe and Reader scaling follow the existing lavinwins/goodpjw2008-derived account recipe. They were not invented or optimized in this analysis.

Reported checkpoint license fields are preserved as provenance, not independently reissued permissions. We do not redistribute upstream weights or code, and no license in this repository grants rights to the competition data or third-party checkpoints.

## Contribution of this repository

Original work here comprises the retrospective study definition, input checks, independent AUC verification, paired bootstrap implementation, branch-removal experiments, class/error summaries, split/provenance audit, reproducible figures and scientific interpretation. This should be credited separately from the original architectures, checkpoints and GPU inference runs.
