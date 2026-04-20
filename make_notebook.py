import json
import os

files = [
    "config.py", "losses.py", "metrics.py", 
    "dataset.py", "attention_unet.py", "transunet.py", 
    "train.py", "evaluate.py", "visualize.py", "main.py"
]

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Comparative Analysis: Attention U-Net vs TransUNet\n",
        "**Skin Lesion Segmentation (ISIC Dataset)**\n\n",
        "This notebook automatically generates our modular PyTorch codebase using `%%writefile` and then runs the entire training and evaluation pipeline end-to-end.\n\n",
        "### Instructions:\n",
        "1. Attach the ISIC 2018 dataset to your Kaggle notebook.\n",
        "2. If your dataset folder isn't named exactly `isic-2018-challenge`, update the `DATA_ROOT` variable in the `config.py` cell below.\n",
        "3. Click **Run All**!"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!pip install albumentations opencv-python-headless scikit-image tqdm scipy matplotlib seaborn\n"]
})

for filename in files:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # If config file, ensure paths are Kaggle compatible (they already are, but just making sure)
    
    lines = content.split('\n')
    source = [f"%%writefile {filename}\n"]
    # Append line with newline character, except the last one if it's empty
    for i, line in enumerate(lines):
        if i == len(lines) - 1 and not line:
            continue
        source.append(line + "\n")
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Run the Complete Pipeline\nThis will train both models, evaluate them, print the metrics, and save figures to the `outputs/figures` folder in your Kaggle working directory."]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!python main.py\n"]
})

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Kaggle_Skin_Lesion_Segmentation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
