# Demand Signal Classification from B2B Sales Notes (Fine-Tuning Phi-1.5 with LoRA)

## Project Overview
This project aims to fine-tune Microsoft's Phi-1.5 pre-trained language model to classify short B2B sales notes into business demand categories.  
We framed this as a multi-class single-label classification task with five classes:
- Repeat Order
- Urgent Need
- Stocking Issue
- Custom Spec
- New Product Demand

We applied Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) to adapt the model to our task while minimizing computational costs.

---

## Dataset Description
Since no public dataset exists for this task, we synthesized a realistic dataset of 2000 examples simulating structured B2B sales notes.  
We introduced minor random noise (e.g., missing spaces, lowercase inconsistencies) to make the data more realistic.

Each sales note is labeled with one of the five demand categories.

---

## Model and Fine-Tuning Strategy
- **Base Model**: `microsoft/phi-1_5`
- **Tokenizer**: GPT-2-style Byte Pair Encoding (BPE) tokenizer, adapted specifically for Phi-1.5.
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: ~1M (only adapter weights)
- **Training Batch Size**: 2 (due to memory limits)
- **Training Epochs**: 3
- **Optimizer**: AdamW
- **Mixed Precision**: Enabled (fp16=True)

We fine-tuned only the LoRA adapters while keeping the base model frozen, following best practices for large model fine-tuning on limited resources.

---

## Evaluation Metrics
- Primary Metric: **Accuracy**
- Additional Metrics: **Precision, Recall, F1-Score (Macro-Average)**
- Confusion Matrix plotted for detailed error analysis.

Evaluation was performed on a 40% held-out validation set.

---

## Results Summary
| Metric | Value |
|--------|-------|
| Final Training Loss | ~0.19 |
| Validation Accuracy | ~100% |
| Notable Observations | Model learned to differentiate urgent needs vs. new product demands quite well |

---

## Folder Structure

```plaintext
/data
    └── sales_notes_2000.csv
/notebooks
    └── 01_data_preprocessing.ipynb
    └── 02_finetune_lora.ipynb
    └── 03_model_training.ipynb
/models
    └── lora_phi15_demand_signal_adapter/ (saved LoRA adapter weights)
/results
    └── (trainer logs, checkpoints, plots)
README.md
requirements.txt
