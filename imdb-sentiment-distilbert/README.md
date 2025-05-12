# IMDb Sentiment Analysis

This project is a simple yet powerful pipeline for binary sentiment classification on IMDb movie reviews using PyTorch and Hugging Face Transformers. It even includes visualizations to help you see what the model pays attention to during inference.

## What’s Inside

- Fine-tuned **DistilBERT** model for classifying movie reviews as positive or negative.
- A balanced dataset: 500 positive and 500 negative reviews, perfect for fast experimentation.
- Modular and easy-to-understand code in the `src/` directory:
  - `data_utils.py` – handles data loading and preprocessing.
  - `dataset.py` – defines a custom PyTorch `Dataset`.
  - `train.py` – training logic for fine-tuning the model.
  - `evaluate.py` – runs evaluation and plots results.
  - `attention_analysis.py` – lets you visualize self-attention weights.
- Interactive Jupyter notebooks in the `notebooks/` folder to help you dive deeper.
- Evaluation results in `results/`.

## Getting Started

Clone the repo and set up your environment:

```bash
git clone https://github.com/ankitsencode123/imdb-sentiment-distilbert.git
cd imdb-sentiment-distilbert
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

