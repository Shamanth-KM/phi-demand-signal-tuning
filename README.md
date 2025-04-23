# phi-demand-intent-lora
LoRA Fine-Tuning of Phi-1.5 to Classify Demand Signals from B2B Sales Notes

# Demand Signal Classification using Phi-1.5 and LoRA

This project fine-tunes Microsoft's Phi-1.5 model using LoRA to classify demand patterns from simulated B2B sales notes. It is designed to support supply chain planning by extracting qualitative demand intent from short-form text.

## Project Structure
- `notebooks/`: Google Colab notebooks for data prep, fine-tuning, and evaluation
- `data/`: Simulated sales notes dataset (CSV)
- `scripts/`: Python modules (e.g., model, training)
- `models/`: Saved LoRA adapters or checkpoints
- `results/`: Evaluation outputs, plots, metrics

## Run on Google Colab
> All notebooks are designed to run in Colab with minimal setup.

## Requirements
Python 3.10 + `transformers`, `peft`, `datasets`, `accelerate`, `scikit-learn`

