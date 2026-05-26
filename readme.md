# Incremental Narrative Generation

Incremental Narrative Generation is a framework for generating pedagogically coherent, multilingual lecture-style narratives from sequential slide content using Large Language Models (LLMs). The repository focuses on discourse-aware educational narration, incremental continuity across slides, pedagogical signal modeling, and evaluation of long-form instructional generation.

---

## Features

### Pedagogical Signal Modeling
Supports discourse-level instructional labels such as:
- Question
- Explanation
- Definition
- Elaboration
- Example
- Recap

### Multilingual Support
Currently supports:
- English
- Hindi
- Bengali

### Evaluation Framework
Includes:
- ROUGE
- METEOR
- BERTScore
- Diversity metrics
- Readability metrics
- Pedagogical consistency metrics
- Narrative continuity analysis

---

## Evaluated Models

The repository currently includes evaluation pipelines for:

- LLaMA 3B
- Gemma 3B
- Qwen 4B
- DeepSeek 1.5B

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Incremental-Narrative-Generation.git
cd Incremental-Narrative-Generation
```

Create environment:

```bash
conda create -n ing python=3.10
conda activate ing
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---
Available evaluations include:

- ROUGE-1 / ROUGE-2 / ROUGE-L
- METEOR
- BERTScore
- Distinct-n
- Self-BLEU
- Contextual relevance
- Dependency completeness
- Pedagogical consistency

---
