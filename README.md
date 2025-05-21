# End-to-End-Breast-Cancer-Classification-Project-Using-DVC-and-MLflow
This project aims to build a robust, reproducible, and version-controlled pipeline for classifying **Invasive Ductal Carcinoma (IDC)** using histopathological image patches. It leverages tools like **DVC** for data and pipeline versioning and **MLflow** for experiment tracking and model management.

---

## Project WorkFlow
The project follows a modular and configurable structure. Here's the high-level workflow:

1.  Update `config.yaml` – Define configurations (paths, URLs, etc.)
2.  Update `secrets.yaml` *(optional)* – For storing sensitive credentials
3.  Update `params.yaml` – Store hyperparameters and other tweakable values
4.  Update entities – Define data structures for consistency
5.  Update Configuration Manager – Load, parse, and validate all configs
6.  Update Components – Implement each step (data ingestion, training, etc.)
7.  Update Pipeline – Chain components into pipelines
8.  Update `main.py` – Entry point to trigger pipeline
9.  Update `dvc.yaml` – Connect stages with DVC for version control

## 📂 II. Dataset

###  Mammography Patch-Based IDC Classification Dataset

####  Overview
**Invasive Ductal Carcinoma (IDC)** is the most common subtype of breast cancer and a key factor in determining tumor aggressiveness. In clinical settings, pathologists identify IDC regions within whole slide images to assess the malignancy grade. For machine learning models aiming to automate this process, isolating these regions becomes a crucial preprocessing step.

This dataset is composed of **patches extracted from 162 whole slide images** of breast cancer tissue specimens, scanned at **40x magnification**. From these slides, a total of **277,524 image patches** (each sized **50 × 50 pixels**) were generated, with binary labels indicating the presence or absence of IDC.

#### Patch Statistics

| Class           | Count     |
|------------------|-----------|
| IDC Negative (0) | 198,738   |
| IDC Positive (1) | 78,786    |

Each image is named using the format:  
`patientID_xX_yY_classC.png`  
Example: `10253_idx5_x1351_y1101_class0.png`

Where:
- `patientID`: Unique identifier for the patient and slide (e.g., `10253_idx5`)
- `xX_yY`: Coordinates indicating where the patch was cropped from
- `classC`: Classification label (`0` = non-IDC, `1` = IDC)

####  Dataset Details
| Attribute        | Description                                |
|------------------|--------------------------------------------|
| **Name**         | IDC Classification Patch Dataset           |
| **Domain**       | Digital Pathology, Breast Cancer Detection |
| **Type**         | Image dataset (histopathology patches)     |
| **Patch Size**   | 50 × 50 pixels                             |
| **Total Samples**| 277,524                                     |
| **Labels**       | Binary (IDC Negative: 0, IDC Positive: 1)  |
| **Image Format** | PNG                                        |

---
### Project Architecture Diagram

```bash
                          ┌────────────────────────────────────┐
                          │     📁 GitHub Repository           │
                          │  - Codebase                       │
                          │  - dvc.yaml                       │
                          │  - requirements.txt               │
                          └────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────┐
                 │   🐍 Python Virtual Env (3.8) │
                 │ - Dependency Management       │
                 └────────────┬──────────────────┘
                              ▼
                    ┌────────────────────┐
                    │   ⚙️ Logging Setup   │
                    └────────────────────┘
                              ▼
              ┌────────────────────────────────┐
              │      🔧 Utility Functions       │
              │ - common.py (Ensure, ConfigBox)│
              └────────────┬───────────────────┘
                           ▼
               ┌───────────────────────────────┐
               │  📥 Data Ingestion (Google Drive) │
               └────────────┬──────────────────┘
                            ▼
                 ┌──────────────────────────────┐
                 │   🧠 Model Preparation         │
                 │ - Base Model + Training       │
                 │ - Evaluation via MLflow       │
                 └────────────┬─────────────────┘
                              ▼
             ┌────────────────────────────────────────┐
             │   📊 MLflow Tracking (Dagshub)          │
             │ - Track Experiments & Metrics          │
             └────────────┬───────────────────────────┘
                          ▼
              ┌────────────────────────────────┐
              │      📦 DVC Pipeline Setup      │
              │ - Versioning Data & Models     │
              └────────────┬───────────────────┘
                           ▼
             ┌────────────────────────────────┐
             │  🔮 Prediction Pipeline (Flask) │
             │ - Deployed via app.py          │
             └────────────┬───────────────────┘
                          ▼
                ┌───────────────────────────┐
                │  🐳 Docker Containerization │
                │ - Dockerfile, Compose      │
                └────────────┬──────────────┘
                             ▼
               ┌────────────────────────────────┐
               │   ⚙️ Jenkins CI/CD on AWS EC2   │
               │ - Automated Testing & Deploy   │
               └────────────────────────────────┘
```

### Citation  
If you use this dataset, please consider citing the original source. Dataset derived from publicly available medical imaging studies.[Mammography Patch-Based IDC Classification Dataset](https://pubmed.ncbi.nlm.nih.gov/27563488/)

---
## Setup Instructions
### Environment Setup
```bash
# Clone repo
git clone https://github.com/your-username/end-to-end-breast-cancer-classification.git
cd end-to-end-breast-cancer-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

## MLflow Integration
MLflow helps track experiments, models, and parameters in real-time.

### MLflow on DAGsHub
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/chest-Disease-Classification-MLflow-DVC.mlflow
MLFLOW_TRACKING_USERNAME=entbappy
MLFLOW_TRACKING_PASSWORD=6824692c47a4545eac5b10041d5c8edbcef0
```

###  Set Environment Variables (Linux/macOS)

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/chest-Disease-Classification-MLflow-DVC.mlflow
export MLFLOW_TRACKING_USERNAME=entbappy
export MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9353c5b10041d5c8edbcef0
```

## DVC Integration

DVC handles version control for data and pipeline stages.

### DVC Commands
```bash
# Initialize DVC
dvc init

# Reproduce pipeline stages
dvc repro

# View pipeline graph
dvc dag

```

## Deployment(CI/CD)
> This project can be extended for AWS deployment using GitHub Actions for CI/CD > > workflows. Add your workflow files under .github/workflows/.


## Acknowldgments
- [Original IDC Dataset Paper](https://pubmed.ncbi.nlm.nih.gov/27563488/)
- [MLflow](https://mlflow.org/) & [DVC](https://dvc.org/) Open Source Communities
- [DAGsHub](https://dagshub.com/) for experiment tracking

## License
This project is licensed under the [Apache-2.0 license](https://github.com/Adity-star/End-to-End-Breast-Cancer-Classification-Using-DVC-and-MLflow?tab=Apache-2.0-1-ov-file#).

