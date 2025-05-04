# End-to-End-Breast-Cancer-Classification-Project-Using-DVC-and-MLflow

## Project WorkFlow
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

## 📂 II. Dataset

###  Mammography Patch-Based IDC Classification Dataset

####  Overview
**Invasive Ductal Carcinoma (IDC)** is the most common subtype of breast cancer and a key factor in determining tumor aggressiveness. In clinical settings, pathologists identify IDC regions within whole slide images to assess the malignancy grade. For machine learning models aiming to automate this process, isolating these regions becomes a crucial preprocessing step.

This dataset is composed of **patches extracted from 162 whole slide images** of breast cancer tissue specimens, scanned at **40x magnification**. From these slides, a total of **277,524 image patches** (each sized **50 × 50 pixels**) were generated, with binary labels indicating the presence or absence of IDC.

- **198,738 patches** labeled as **IDC Negative**
- **78,786 patches** labeled as **IDC Positive**

Each image file follows a consistent naming format: patientID_xX_yY_classC.png

Example:  
`10253_idx5_x1351_y1101_class0.png`

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

### Citation  
If you use this dataset, please consider citing the original source. Dataset derived from publicly available medical imaging studies.[Mammography Patch-Based IDC Classification Dataset](https://pubmed.ncbi.nlm.nih.gov/27563488/)

---




