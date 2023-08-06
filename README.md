# MBIT thesis: Explainable AI in Job Recommendation Systems (TO BE UPDATE)
Graduation thesis project: Title: Explainable AI in Job Recommendation System <br>
Data: [Kaggle's CareerBuilder 2012](https://www.kaggle.com/c/job-recommendation) <br>
**Requirements: important libraries**

- sklearn, pandas, numpy
- [myfm](https://myfm.readthedocs.io/en/stable/)
- [interpretml](https://interpret.ml/docs/getting-started)
- [SHAP](https://shap-lrjball.readthedocs.io/en/latest/index.html)
- [LIME](https://lime-ml.readthedocs.io/en/latest/)

Apart from public libraries, please import modules in [utils](https://github.com/anhtth16/ut_mbit_thesis/tree/main/utils) folder for generating recommmendations

High-resolution figures in the report: [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/figures_overleaf)

# Navigation guidelines:

CHECK LIST (based on Table Of Content) of the report (hyperlink to the repo)
### 4.1: Data pre-processing and Feature Engineering: [folder](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_data_prep)
**PENDING OUPTUT (Large dataset)**

- 4.1.1 Data cleaning 
- 4.1.2 Data augmentation: Negative sampling for interaction data - [link](https://github.com/anhtth16/ut_mbit_thesis/blob/main/nb_data_prep/negative_sampling.ipynb)

Feature Engineering:

TFIDF for both jobs and user history - [link](https://github.com/anhtth16/ut_mbit_thesis/blob/main/nb_data_prep/feature_engineering_tfidf.ipynb) 

- 4.1.4 Feature Engineering: Generate location matching features 
- 4.1.5 Feature Engineering: Transform text features

LDA for jobs - [link](https://github.com/anhtth16/ut_mbit_thesis/blob/main/nb_data_prep/lda_jobs.ipynb), LDA for user history - [link](https://github.com/anhtth16/ut_mbit_thesis/blob/main/nb_data_prep/lda_users.ipynb)

- 4.1.6 Feature Engineering: Discretizing user profile features: [link](https://github.com/anhtth16/ut_mbit_thesis/blob/main/nb_data_prep/discretize_data.ipynb)

### 4.2: Generating potential applications: [folder](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_ranking_data)

- 4.2.1 Potential application generation by random sampling with control on positive label
- 4.2.2 Potential application generation by unsupervised KNN models (2 variations: knn\_lda, knn\_tfidf)

### 4.3: Training ranking models:
**PENDING OUPTUT (Large pre-trained models - FM)**

- White-Box, Black-box models: 7 models [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_baseline_tabular), [pre-trained models](https://github.com/anhtth16/ut_mbit_thesis/tree/main/output_baseline_tabular)
- Factorization Machine models: 4 models [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_myfm), [output - LARGE FILE pre-trained models]()
- Explanable Boosting Machine models: 3 EBM models and 3 DP-EBM models [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/xai_recsys)

### 4.4: Finalize ranking results and evaluation JRS

Generate top 20 recommendation: 20 jobs/ user <br>
Output format: UserID, JobID, Y\_pred, Y\_prob, rank <br>
(Y\_pred: predicted label, Y\_prob: probability of prediction, rank: ranking based on probability)

Each model have 2 potential sources of application.

- White-box & Black-box recsys: [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_recsys_tabular), [output](https://github.com/anhtth16/ut_mbit_thesis/tree/main/output_topN_tabular)
-  FM recsys: [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_recsys_fm), [output](https://github.com/anhtth16/ut_mbit_thesis/tree/main/output_topN_myfm)
- EBM & DPEBM recsys: [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_recsys_ebm), [output](https://github.com/anhtth16/ut_mbit_thesis/tree/main/output_topN_ebm)

### 4.5: Explaining recommendations:
- 4.5.2 Global explanation by model-specific approach: EBM models [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_self_explanation)
- 4.5.3 Global explanation by model-specific approach: DPEBM models [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_self_explanation)
- 4.5.4 Global self-explanation by white-box models and XGBoost [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_self_explanation)
- 4.5.5 KernelSHAP: Local feature importance [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/xai_posthoc), [output](https://github.com/anhtth16/ut_mbit_thesis/tree/main/output_shap)
- 4.5.6 LIME: Localfeature importance [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/xai_posthoc), [output](https://github.com/anhtth16/ut_mbit_thesis/tree/main/output_lime)

### 4.6: Evaluation explanation

- 4.6.1 Modelfidelity rate-Global explanation [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_xai_fidelity)
- 4.6.2 Feature importance fidelity rate - Local explanation  [link](https://github.com/anhtth16/ut_mbit_thesis/tree/main/nb_xai_fidelity)

### 4.7 Generating human-digestible explanation
- 4.7.2 Usecase Post-explanation: extract raw term from TF-IDF features [link](https://github.com/anhtth16/ut_mbit_thesis/blob/main/nb_xai_viz/usecase_viz_logreg_explanation.ipynb)
- 4.7.3  Usecase Post-explanation: LDA topic contribution visualization: [link](https://github.com/anhtth16/ut_mbit_thesis/blob/main/nb_xai_viz/usecase_viz_ebm_explanation.ipynb)

### Additional Analysis [pending]

# Reproducing guidelines:
