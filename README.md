<p align="center">
<img width="300" src="https://raw.githubusercontent.com/Raminghorbanii/PATE/master/docs/PATE_logo.png"/>
</p>


<h1 align="center">PATE: Proximity-Aware Time series anomaly Evaluation</h1>

<p align="center">
  <a href="https://kdd.org/kdd2024/">
    <img src="https://img.shields.io/badge/KDD-2024-blue.svg" alt="KDD 2024 Accepted">
  </a>
</p>

This repository contains the code for PATE (Proximity-Aware Time series anomaly Evaluation measure), a novel evaluation metric for assessing anomaly detection in time series data. PATE introduces proximity-based weighting and computes a weighted version of the area under the Precision-Recall curve, offering a more accurate and fair evaluation by considering the temporal relationship between predicted and actual anomalies. The methodology is detailed in our paper, showcasing its effectiveness through experiments with both synthetic and real-world datasets.
 

## Quick Start

### Installation
Install PATE for immediate use in your projects:

```bash
pip install PATE
```

## How to use PATE? 
Utilizing PATE is straightforward. Begin by importing the PATE module in your Python script:

```bash
from pate.PATE_metric import PATE
```

Prepare your input as arrays of anomaly scores (continues or binary) and binary labels. PATE allows for comprehensive customization of parameters, enabling easy toggling between PATE and PATE-F1 evaluations. Please refer to the main code documentation for a full list of configurable options.

Example usage of PATE and PATE-F1:

```bash
pate = PATE(labels, anomaly_scores, binary_scores = False)
pate_f1 = PATE(labels, binary_anomaly_scores, binary_scores = True)
```

### Basic Example

```python 
import numpy as np
from pate.PATE_metric import PATE

# Example data setup
labels = np.array([0, 1, 0, 1, 0])
scores = np.array([0.1, 0.8, 0.1, 0.9, 0.2])

# Initialize PATE and compute the metric
pate = PATE(labels, scores, binary_scores = False)
print(pate)
```

---

## Advanced Setup and Experiments
For researchers interested in reproducing the experiments or exploring the evaluation metric further with various data sets:


### Environment Setup
To use PATE, start by creating and activating a new Conda environment using the following commands:

```bash
conda create --name pate_env python=3.8
conda activate pate_env
```

### Install Dependencies
Install the required Python packages via:

```bash
git clone https://github.com/raminghorbanii/PATE
cd PATE
pip install -r synthetic_exp_requirements.txt
```

## Conducting Experiments

### with Synthetic Data

To run experiments on synthetic data, navigate to the experiments/Synthetic_Data_Experiments directory and execute the main Python script.
This script allows for the modification of various scenarios, comparing PATE and PATE-F1 against other established metrics.


```bash
cd experiments/Synthetic_Data_Experiments
python main_synthetic_data.py
```

Example of how you use PATE using synthetic data (Binary detector):

```python

from utils_Synthetic_exp import evaluate_all_metrics, synthetic_generator

label_anomaly_ranges = [[40,59]] # You can selec multiple ranges for anomaly. Here we selected one range with the size of 20 points (A_k) 
predicted_ranges = [[30, 49]]  # You can selec multiple ranges for predictions. Here we selected the range the same as Scenario 2, proposed in the original paper. 
vus_zone_size = e_buffer = d_buffer = 20 

experiment_results = synthetic_generator(label_anomaly_ranges, predicted_ranges, vus_zone_size, e_buffer, d_buffer)
predicted_array = experiment_results["predicted_array"]
label_array = experiment_results["label_array"]


score_list_simple = evaluate_all_metrics(predicted_array, label_array, vus_zone_size, e_buffer, d_buffer)
print(score_list_simple)


```


```bash

Output:

'original_F1Score': 0.5,
'pa_precision': 0.67,
'pa_recall': 1.0,
'pa_f_score': 0.8,
'Rbased_precision': 0.6,
'Rbased_recall': 0.6,
'Rbased_f1score': 0.6,
'eTaPR_precision': 0.75,
'eTaPR_recall': 0.75,
'eTaPR_f1_score': 0.75,
'Affiliation precision': 0.97,
'Affiliation recall': 0.99,
'Affliation F1score': 0.98,
'VUS_ROC': 0.79,
'VUS_PR': 0.72,
'AUC': 0.74,
'AUC_PR': 0.51,

'PATE': 0.76,
'PATE-F1': 0.75}

```

### with Real-World Data
For real-world data experiments, ensure all additional required packages are installed.

```bash
pip install -r Real_exp_requirements.txt
```

#### Download the Dataset
The datasets for these experiments can be downloaded from the following link:

Dataset Link: https://www.thedatum.org/datasets/TSB-UAD-Public.zip 

Ref: This dataset is made available through the GitHub page of the project "An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection (TSB-UAD)": https://github.com/TheDatumOrg/TSB-UAD

#### Running the Experiments

After downloading, place the unzipped dataset in the same directory. If you store the data in a different location, ensure you update the directory paths in the code to match.

Navigate to the experiments/RealWorld_Data_Experiments directory to run an experiment. Execute one of the example Python scripts by entering the following command:

```bash
cd experiments/RealWorld_Data_Experiments
python Example1.py
```
Two different examples are provided. These examples allow for modifications and customizations, enabling detailed exploration of various data aspects.


---

## Setting Buffer Size in PATE

Given the context of time series data, selecting a buffer size for a fair evaluation of anomaly detectors' performance is unavoidable. The buffer parameter of PATE can be set using the following strategies:

- *Expert Knowledge*: Best suited for customized, specific, and real-world applications where expert knowledge is available, or when one has enough experience with the data at hand. Experts can directly specify buffer sizes that are optimized for the particular use case.

- *ACF Analysis*: Automatically determines the optimal buffer size by analyzing the autocorrelation within the data. This function is available in PATE_utils.py.

- *Range of Buffer Sizes*: PATE is flexible and can evaluate performance across all combinations of pre and post buffer sizes, allowing for a comprehensive assessment without expert input. One can start with a maximum buffer size, and PATE automatically divides it into a specified number of ranges (determined by the user).

- *Default Setting*: Utilizes the input window size of the anomaly detector, a standard, practical buffer size that aligns with the general scale of the data being analyzed. This option is useful when no specific adjustments are needed or when minimal configuration is desired.

This guidance ensures that you can effectively implement these buffer size selection strategies in PATE for optimal results.


---

## Citation

If you use PATE in your research or in any project, we kindly request that you cite the PATE paper:

```bash

```