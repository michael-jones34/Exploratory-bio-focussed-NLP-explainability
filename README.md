## Explaining Natural Language Processing Predictions in Biosecurity Text Classification

**Abstract**

This project explores model explainability in NLP by comparing how different explainability methods behave when applied to both simple and complex text classification models. Using a manually labelled dataset of biomedical abstracts, I train a Bag-of-Words logistic regression model and a fine-tuned BioBERT model to predict whether an abstract is pathogen-related. I then apply SHAP, LIME, and gradient-based attribution methods to analyse what each model learns and how intuitive and stable the resulting explanations are. The results show that increased model complexity does not necessarily yield better performance or more meaningful explanations, highlighting important tradeoffs between accuracy, explainability, and computational cost in high-stakes domains such as biosecurity.

**Full technical report**

[Full report](report/report.md)

**Contents**

data/         – Input dataset (not included)  
models/       – Trained models (BoW included, BioBERT excluded)  
notebooks/    – Main analysis and experiments  
report/       – Full written report and figures  
results/       - Results used for figures and tables

**Environment**

- Python 3.12
- Key libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn, shap, lime, captum, torch, transformers, joblib, IPython

A virtual environment (.venv) is recommended. A requirements.txt is provided, though some packages may require manual installation depending on platform and Python version.

**Data**

Place the labelled dataset at:

`data/Pathogen_abstracts_gold_dataset.csv`

Required columns: `id`, `title`, `abstract`, `label`

**Models**
- BoW model: included in `models/BoW/`
- BioBERT model: not included due to size (~413 MB)

To run BioBERT-related cells, download the model from:  
https://drive.google.com/file/d/1cGQbbr381Di96oxDf_0dePD5axTVFrc8/view?usp=sharing  
and place it in:  
`models/BioBERT/`

**Running the notebook**

Open:

`notebooks/abstracts_recognition.ipynb`  

Run cells top to bottom.
BioBERT and Captum sections are computationally expensive.

**Scope note**

This repository is intended as an exploratory research and learning project rather than a production-ready system. The models and explanations are evaluated on a small, manually labelled dataset and should not be used for real-world decision-making without significant extension and validation.

