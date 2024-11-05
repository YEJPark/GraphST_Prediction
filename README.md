# GraphST_Prediction

**GraphST_Prediction** is a toolkit for applying graph-based contrastive learning methods to spatial transcriptomics data at subcellular resolution. This repository includes scripts for data preprocessing, model training, and evaluation, specifically designed for predictive tasks in spatial transcriptomics.

## Repository Structure
GraphST_Prediction/
├── src/                           # Source code and supporting modules
│   ├── preprocessing/             # Data preprocessing scripts
│   │   ├── add_cancer_column.py   
│   │   ├── merge_fov_files.py     
│   │   ├── merge_fov_selected_columns.py 
│   │   ├── process_cell_files.py  
│   │   └── process_fov_files.py   
│   ├── data_loader.py             # Data loading and preparation functions
│   └── utils.py                   # Utility functions used across scripts
├── main_argmax.py                 # Main script for argmax-based model training
├── main_sigmoid.py                # Main script for sigmoid-based model training
├── main_softmax_argmax.py         # Main script for softmax + argmax model training
├── README.md                      # Repository documentation and instructions
└── requirements.txt               # Package requirements file

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/GraphST_Prediction.git
cd GraphST_Prediction

### 2.Install dependencies
Use the requirements.txt file to install all necessary packages.
pip install -r requirements.txt





---

This is the complete README content in one code block, ready to copy and paste directly into your `README.md` file. Just replace `YOUR_USERNAME` with your GitHub username. Let me know if you need further assistance!

