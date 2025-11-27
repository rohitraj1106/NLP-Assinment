# Zero‑Shot & Fine‑Tuned News Classification (NLP Assignment)

This repository contains two NLP mini‑projects built as part of an academic assignment:  
1) Zero‑shot news classification using a pretrained transformer model, and  
2) Supervised/fine‑tuned text classification on a news dataset. [attached_file:1][attached_file:2]

Both projects are implemented in Python using Hugging Face Transformers, Datasets, and common ML tools. [attached_file:1][attached_file:2]

## Project Structure

- `nlp-project-1.ipynb` – Supervised news classifier with data loading, preprocessing, model training, evaluation, and basic analysis. [attached_file:1]  
- `nlp-project-2.ipynb` – Zero‑shot news classifier on the AG News dataset with a small Gradio demo app. [attached_file:2]  

You can optionally export the notebooks to `.py` scripts if needed. [attached_file:1][attached_file:2]

## Features

- Transformer‑based models from Hugging Face (e.g. `facebook/bart-large-mnli` for zero‑shot classification). [attached_file:2]  
- AG News dataset loaded via `datasets.load_dataset` for fast experimentation. [attached_file:2]  
- Evaluation with accuracy and a detailed classification report (precision, recall, F1). [attached_file:1][attached_file:2]  
- Interactive Gradio web demo for trying custom news headlines with the zero‑shot model. [attached_file:2]

## Installation

Clone the repository and install dependencies (Python 3.10+ recommended):
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install -r requirements.txt

If you do not have a `requirements.txt` yet, create one with at least: [attached_file:1][attached_file:2]
transformers
datasets
evaluate
torch
scikit-learn
matplotlib
seaborn
streamlit
gradio
tqdm
pandas

GPU acceleration is supported if you install a CUDA‑enabled version of PyTorch; otherwise the code runs on CPU. [attached_file:2]

## Usage

### 1. Supervised News Classification (`nlp-project-1.ipynb`)

1. Open `nlp-project-1.ipynb` in Jupyter, JupyterLab, or VS Code. [attached_file:1]  
2. Run all cells to:
   - Install and import required libraries.  
   - Load and preprocess the dataset.  
   - Train the classifier and print accuracy and the classification report. [attached_file:1]  

Configuration such as model name, batch size, and epochs can be edited in the configuration cells. [attached_file:1]

### 2. Zero‑Shot News Classification + Gradio App (`nlp-project-2.ipynb`)

1. Open `nlp-project-2.ipynb`. [attached_file:2]  
2. Run the notebook cells in order to:
   - Install and import Transformers, Datasets, scikit‑learn, and Gradio.  
   - Load a subset of AG News (e.g. 100 samples) and evaluate the zero‑shot classifier (about 0.76 accuracy on the sampled subset). [attached_file:2]  
   - Launch a Gradio interface that classifies custom news text into  
     `World Politics`, `Sports`, `Business`, or `Science and Technology`. [attached_file:2]  

The Gradio app prints a local URL and an optional public sharing URL in the cell output when launched. [attached_file:2]

## Dataset and Labels

- Dataset: AG News (4‑class topic classification for news headlines/articles). [attached_file:2]  
- Label mapping used in the assignment:  
  - `0 → World Politics`  
  - `1 → Sports`  
  - `2 → Business`  
  - `3 → Science and Technology`  
[attached_file:2]

## Evaluation

For the sampled AG News subset in the zero‑shot notebook: [attached_file:2]  

- Accuracy is around 0.76 on 100 examples.  
- `sklearn.metrics.classification_report` prints per‑class precision, recall, and F1‑score.  

The supervised model notebook prints similar metrics after training on its dataset. [attached_file:1]

## Reproducibility and Configuration

- Random seeds such as `seed=42` are set when shuffling/selecting subsets to keep splits deterministic across runs. [attached_file:2]  
- Key options (model name, sample size, device CPU/GPU) are configured near the top of each notebook and can be modified easily. [attached_file:1][attached_file:2]

## Acknowledgements

- Pretrained models and tokenizers from [Hugging Face Transformers](https://huggingface.co/transformers/). [attached_file:1][attached_file:2]  
- Datasets via the [Hugging Face Datasets](https://huggingface.co/docs/datasets) library (AG News). [attached_file:2]  
- Web UI built with [Gradio](https://www.gradio.app/). [attached_file:2]

