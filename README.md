# Child Mind Institute — Problematic Internet Use  
**Predicting Problematic Internet Usage in Youth with Activity and Survey Data**

## 📝 Project Overview
This repository contains code and research for the [Kaggle Child Mind Institute — Problematic Internet Use competition]. The challenge is to **predict the Severity Impairment Index (sii)** for children and adolescents, using a mix of actigraphy (wearable device activity signals), biometric data, clinical scales, and self/parent questionnaires.[1]
The target, **sii**, is a 4-class problem (None, Mild, Moderate, Severe), derived from endorsed internet addiction symptoms.

***

## 📊 Data Sources
The competition dataset is based on the Healthy Brain Network (HBN), sampling youth aged 5–22.  
- **Tabular data**: `train.csv`, `test.csv` — demographic, fitness, questionnaire, clinical and behavioral metrics
- **Series data**: `series_train.parquet`, `series_test.parquet` — up to 30 days of continuous wrist accelerometer data per participant
- **Data dictionary**: `data_dictionary.csv` — instrument and feature description
- **Sample submission**: `sample_submission.csv`

Key features used in modeling:
- **Physical Activity and FitnessGram**: BMI, cardiovascular, endurance, body composition, etc.
- **Internet Use**: Self/parent-reported daily use and PCIAT results
- **CGAS**: Mental health functioning
- **Sleep and Clinical Assessments**
- **Actigraphy statistics**: ENMO, angle, light, time series moments

***

## ⚙️ Model Pipeline  
This project includes:
- **Actigraphy Processing**: Loads per-subject .parquet files, computes descriptive statistics
- **Data Preprocessing**: Merged tabular and latent time-series representations; imputation with KNN
- **Feature Engineering**: Creation of latent sequence encoding using LSTM neural net
- **Categorical Data Handling**: One-hot or ordinal encoding of instrument seasonality and category fields
- **Model Training**: Voting ensemble (LightGBM, XGBoost, CatBoost) for multi-class regression; optimal thresholds mapped to sii classes
- **Submission Generation**: Output formatted for direct Kaggle leaderboard submission

Example workflow:
- Extract time-series stats → Build LSTM encoder for latent features  
- Merge latent features with tabular features  
- Impute missing values  
- Encode categorical variables  
- Train ensemble regressor and tune thresholds for best QWK score  
- Output submission file

***

## 🔧 Usage Instructions

### 1. Clone the repository  
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
### 2. Install dependencies  
See the section below for recommended packages.  
```bash
pip install -r requirements.txt
```
### 3. Download competition data  
Manually download all files from the [Kaggle competition data page] and place them in the `data/` folder.[1]

### 4. Run code  
Start with the provided Jupyter notebook (`lstm-code.ipynb`). This notebook shows step by step:
- Data loading and merging
- Actigraphy feature extraction
- Latent feature encoding with LSTM
- Tabular/latent feature merging and model training
- Submission formatting

***

## 🧰 Requirements

Typical libraries for this stack are below (see inside your notebook for specifics):
```
pandas
numpy
scikit-learn
matplotlib
lightgbm
xgboost
catboost
tqdm
torch
keras
colorama
```

***

## 📂 Directory Structure

```
.
├── data/                 # Downloaded Kaggle .csv, .parquet files
├── lstm-code.ipynb       # Main notebook for raw data, LSTM encoding, model training
├── README.md
├── requirements.txt      # List of required Python libraries
└── submission.csv        # Generated submission to Kaggle
```

***

## 🚀 Technical Highlights

- **Multi-Modal Feature Integration**: Combines wearable accelerometer statistics (actigraphy) with clinical, self/parent questionnaires, demographic, and biometric data to capture both objective and subjective factors linked to problematic internet use.[1][2]
- **Efficient Time-Series Processing**: Parallelized loading and summarization of per-participant accelerometer .parquet files, turning high-frequency raw signals into interpretable latent descriptors that capture personal activity trends.[3]
- **LSTM-Based Latent Encoding**: Custom PyTorch LSTM encoder compresses multiday actigraphy statistics into rich, trainable latent vectors, enabling ensemble models to leverage sequential physical activity as predictors for internet addiction severity.[4][3]
- **Robust Imputation and Preprocessing**: KNN- and categorical imputation fills gaps in tabular data, essential given clinic data sparsity and heterogeneous survey completion across samples.[3]
- **Automated Threshold Optimization**: Custom QWK-based postprocessing finds thresholds for mapping regressor outputs to discrete SIIs, maximizing clinical utility and leaderboard scores.[5][3]
- **Voting Ensemble Model**: XGBoost, LightGBM, and CatBoost models are ensembled for robust multi-class classification, balancing depth and feature interactions for generalization.[5][3]

***

## 🌍 Project Impact

- **Advancing Early Detection**: By linking physical activity patterns and fitness biosignatures to screen time issues, the project provides a scalable path for clinicians and researchers to identify youth at risk for problematic internet use, even before overt behavioral symptoms or drop in academic/social functioning.[6][2]
- **Reducing Barriers in Mental Health**: Uses widely accessible, non-invasive data sources (wearables, fitness/health surveys) for screening, making population-level risk assessments feasible outside of specialist clinics.[2][7]
- **Accelerating Open Science Collaboration**: The Healthy Brain Network dataset is shared globally, supporting more than 490 papers from hundreds of teams and facilitating reproducible, generalizable advances in mental health and developmental research.[8][7][9]
- **Population-Level Health Insights**: The competition has generated tools and insights for clinicians, educators, and families, directly informing strategies to combat digital overuse and promote healthier habits in youth, benefiting both direct participants and future generations.[6][2]

***

## 📜 License
Repository is released under MIT.  
Data is for competition and research only — see Kaggle terms for details.
