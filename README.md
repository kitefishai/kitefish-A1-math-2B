# ArXiv-Logic SLM: A 2B Parameter Scientific Reasoner

An experimental 2 billion parameter Small Language Model (SLM) trained from scratch on a curated 160GB+ mixture of LaTeX, code, and scientific reasoning data. This project focuses on pushing the boundaries of mathematical reasoning and formal proof generation in small-scale architectures.

## 🚀 Overview
Most small models struggle with the "reasoning gap." This model is architected to bridge that gap by prioritizing logic-heavy datasets (GitHub Code, Formal Proofs) alongside a massive corpus of peer-reviewed scientific literature (arXiv).

### Key Features:
* **Parameter Count:** 2.0B
* **Primary Focus:** Mathematics, Physics, Computer Science.
* **Architecture:** Llama-3 style decoder-only transformer.
* **Capabilities:** Multi-step LaTeX derivation, code-to-logic translation, and RAG-ready conversational behavior.

---

## 📊 Data Mixture & Sampling Strategy
To ensure the model clears high-level benchmarks (AIME, MATH), we utilized a "Gold-Standard" sampling strategy. We up-sampled high-density reasoning data to ensure logical stability.

| Dataset Source | Weight | Volume | Purpose |
| :--- |:-------|:-------| :--- |
| **Cleaned arXiv (Gold)** | 1.0x   | 80 GB  | Specialized Domain Expertise |
| **OpenWebMath** | 2.0x   | 50 GB  | General Mathematical Concepts |
| **StackExchange** | 2.0x   | 10 GB  | Q&A Reasoning & Community Logic |x
| **MathInstruct** | 1.0x   | 260k   | Multi-step Problem Solving |
| **PubMedQA / SciQ** | 1.0x   | 50k    | Scientific Verification |
| **UltraChat** | 1.0x   | 50k    | RAG & Conversational Alignment |



---

## 🧪 Evaluation Benchmarks
The model is evaluated against a suite of specialized STEM and Reasoning benchmarks:

### Mathematical Reasoning
* **MATH:** Level 1-5 competition math.
* **GSM8K:** Multi-step grade school math.

### Scientific Knowledge
* **PubMedQA:** Medical question answering.
* **SciQ:** Science exam questions (Physics, Chemistry, Biology).

---

## 🛠️ Getting Started

### Prerequisites
* **Storage:** ~200GB for full dataset; 8GB for model weights.
* **Hardware:** Optimized for 2x A100 (80GB) training.

### Installation
```bash
git clone [https://github.com/your-username/arxiv-logic-slm.git](https://github.com/your-username/arxiv-logic-slm.git)
cd arxiv-logic-slm
pip install -r requirements.txt