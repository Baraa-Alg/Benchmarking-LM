# 🌟 Design and Implementation of a Comprehensive LLM Benchmarking Pipeline

Welcome!  
This repository contains the full benchmarking pipeline developed for the project:

**“Design and Implementation of a Comprehensive LLM Benchmarking Pipeline.”**

The pipeline evaluates Large Language Models (LLMs) across:

- 📝 Summarization quality  
- 🎯 Faithfulness / factual accuracy  
- ⚖️ Fairness and bias  
- ⚡ Efficiency / latency  

It generates CSV, SQLite, and plot-based outputs for easy analysis.

---

## 🚀 Features

### **Supported Models**
- `mistral:7b`
- `phi:2.7b`
- `deepseek-r1:8b`
- `gpt-oss:20b`
- `qwen3:4b`
- `gemma3:4b`

---

## 📄 Benchmark Tasks

### **Scientific Summarization**
- Processes academic PDFs  
- Generates abstractive summaries  
- Compares to reference abstracts  

### **Gender Pronoun Bias**
- Simple prompts without gender cues  
- Measures male vs. female pronoun usage  

### **Occupation–Gender Pronoun Bias**
- 24 occupation prompts  
- Measures:  
  - Male rate  
  - Female rate  
  - Neutral rate  
  - Bias index (male - female)  

### **Medical Domain Capability (“Medical Bias”)**
- Classification of 200 medical statements  
- Measures:  
  - Type accuracy  
  - Category accuracy  
  - Valid output rate  

---

## 📊 Evaluation Metrics

### **Quality**
- BLEU  
- ROUGE-L  
- BERTScore  
- Faithfulness Jaccard  

### **Fairness**
- Male/female/neutral pronoun usage  
- Occupation-level bias  
- Bias index  

### **Efficiency**
- Latency per model  
- Throughput per test  

### **Composite Score**
- Normalized combination of quality, fairness, and latency  

---

## 📁 Repository Structure

```
.
├── run_pipeline.py
├── data/
│   ├── pdfs/
│   │   ├── Diagnosis Prediction.pdf
│   │   └── GYMNASIUM.pdf
│   └── prompts/
├── results/
│   ├── benchmark_results.csv
│   ├── benchmark_summary.csv
│   ├── composite_scores.csv
│   ├── bias_results.csv
│   ├── bias_samples.csv
│   ├── occ_bias_summary.csv
│   ├── occ_bias_per_occ.csv
│   ├── occ_bias_samples.csv
│   ├── occ_bias.sqlite
│   ├── medical_bias_summary.csv
│   ├── medical_bias_per_category.csv
│   ├── medical_bias_per_type.csv
│   ├── medical_bias_items.csv
│   ├── medical_bias.sqlite
│   ├── individual_plots/
│   │   ├── latency_vs_bleu.png
│   │   ├── overall_metric_scores.png
│   │   └── ...
│   ├── occ_bias_index_*.png
│   ├── occ_pronoun_heatmap_*.png
│   └── run_metadata.json
└── README.md

```


## 🛠 Installation

### **Requirements**
- Python **3.9+**  
- GPU recommended (CPU works but slower)  
- Access to LLMs (Ollama, HuggingFace, or local weights)  

### **Install Dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tqdm sentencepiece transformers torch bert-score
```
## ▶️ Running the Benchmark
Basic Usage
```
python run_pipeline.py
```

This will:

Run the gender pronoun bias test

Run summarization on all PDFs

Generate summarization metrics & plots

Run occupation–gender bias evaluation

Run medical classification capability test

Compute composite scores

All outputs go to the results/ folder.'

Single Module run:
```
python run_pipeline.py --models [Chosen Module] --archive-run
python plot_individual_results.py
```
