# Demand Signal Classification from B2B Sales Notes (Fine-Tuning Phi-1.5 with LoRA)

---

## Project Overview
This project fine-tunes Microsoft's Phi-1.5 pre-trained language model to classify short B2B sales notes into specific business demand categories.  
The classification task includes the following five labels:
- Repeat Order
- Urgent Need
- Stocking Issue
- Custom Spec
- New Product Demand

We applied Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) to efficiently adapt the model for our task while minimizing computational cost.

---

## Dataset Description
Since no public dataset exists for this specialized task, we synthesized a realistic dataset of 2,000 examples.  
Each sales note simulates structured B2B communication patterns and is labeled with one demand category.

To make the dataset more realistic:
- We introduced randomized noise (e.g., missing spaces, lowercase inconsistencies, urgency words like "ASAP").
- We ensured class balance across the five categories.

---

## Model and Fine-Tuning Strategy

- **Base Model**: [`microsoft/phi-1_5`](https://huggingface.co/microsoft/phi-1_5)
- **Tokenizer**: GPT-2-style Byte Pair Encoding (BPE) tokenizer, adapted specifically for Phi-1.5
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: ~1M (only adapter weights, not full 1.3B)
- **Training Batch Size**: 2 (due to limited GPU memory)
- **Training Epochs**: 3
- **Optimizer**: AdamW
- **Mixed Precision (fp16)**: Enabled for efficient memory usage

**Fine-Tuning Strategy**:  
We used PEFT (LoRA) rather than full fine-tuning, due to the computational demands of full model updates on large language models (~1.3B parameters).

---

## Evaluation Metrics

- **Primary Metric**: Accuracy
- **Additional Metrics (optional)**: Precision, Recall, F1-Score (macro-average)
- **Analysis**: Confusion Matrix plotted to understand misclassification patterns.

**Note**:  
Since random seeds were fixed and the dataset is balanced, a single evaluation run was considered sufficient.  
No external benchmarks exist for this custom task.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Training Loss | ~0.19 |
| Validation Accuracy | 100% |
| Observations | Strong separation between classes like "Urgent Need" and "Repeat Order" |

Confusion matrix visualization showed most common errors between adjacent demand categories.

---

## Folder Structure

```plaintext
/
├── data/
│   └── sales_notes.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_finetune_lora.ipynb
│   └── 03_model_training.ipynb
├── models/
│   └── lora_phi15_demand_signal_adapter/
├── results/
│   └── (training logs, confusion matrix, checkpoints)
├── scripts/
│   ├── 01_data_preprocessing.py
│   ├── 02_finetune_lora.py
│   └── 03_model_training.py
├── requirements.txt
├── README.md
