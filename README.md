# NLP_MAP

**MIRAGE:** Misconception Detection with Retrieval-Guided Multi-Stage Reasoning and Ensemble Fusion

This repository contains the official implementation of MIRAGE, a novel, cost-effective hybrid framework for automated student misconception detection in mathematics, presented at SOICT 2025.

The project addresses the challenge of analyzing open-ended student responses by integrating retrieval-augmented generation (RAG) principles with Chain-of-Thought (CoT) reasoning and ensemble learning.

---

## 🌟 Overview

Detecting student misconceptions in open-ended responses requires both semantic precision and complex logical reasoning. MIRAGE is designed to provide a scalable and effective solution by reducing reliance on large-scale language models (LLMs) during inference while maintaining high predictive accuracy.

The framework operates in three primary stages:

1. **Retrieval:** Narrows down a large pool of possible error types to a semantically relevant subset.
2. **Reasoning (CoT):** Employs structured Chain-of-Thought generation to expose logical inconsistencies in student solutions.
3. **Reranking:** Refines predictions by aligning candidate labels with the generated reasoning traces.

These components are unified through a **Fusion Ensemble Mechanism** to enhance robustness, accuracy, and interpretability.

---

## 📐 Architecture

MIRAGE is structured as a cohesive pipeline that integrates multiple specialized language models (LMs) through a fusion strategy.

| Module             | Purpose                                                                 | Key Technique                               | Model Used (Example) |
|-------------------|-------------------------------------------------------------------------|--------------------------------------------|--------------------|
| Retrieval Module   | Identifies top-k candidate misconception labels based on semantic similarity to the input query (Q, A, E). | Masked Supervised Contrastive Loss ($\mathcal{L}_{MaskSupCon}$) | MathBERT (Embedder) |
| Reasoning Module   | Generates detailed, step-by-step reasoning ($R$) to analyze logical inconsistencies in the student's explanation. | Knowledge Distillation (from CoT Teacher LLM) | Qwen3-8B (Reasoner) |
| Reranking Module   | Re-scores candidates by evaluating their consistency with the query and the generated reasoning ($R$). | Verification-based Reranking (Logit Margin) | Qwen3-7B (Reranker) |
| Fusion Ensemble    | Aggregates the normalized scores from the Retrieval and Reranking modules using a weighted fusion strategy ($\alpha, \beta$). | Weighted Fusion | N/A |

---

### Key Contributions

- **Hybrid Two-Stage Framework:** Combines similarity-based retrieval and a cross-attention reranker conditioned on the question, student answer, and reasoning.
- **CoT Integration & Interpretability:** Uses Chain-of-Thought (CoT) to generate intermediate explanations, guiding classification and improving model transparency.
- **Verification-based Reranking:** Reformulates reranking as a verification check using a logit-difference scheme.
- **Knowledge Distillation:** Transfers knowledge from a Teacher LLM (GPT-OSS-20B) into smaller, cost-effective models (Qwen3-8B) to retain performance while reducing inference overhead.

---

## 🚀 Performance

Evaluated on the MAP Student Misconceptions dataset:

| Method                 | MAP@1 | MAP@3 | MAP@5 |
|------------------------|-------|-------|-------|
| Reranking Module (Baseline) | 0.79  | 0.81  | 0.88  |
| Retrieval Module (Baseline) | 0.74  | 0.83  | 0.85  |
| MIRAGE (Ensemble)      | 0.82  | 0.92  | 0.93  |

The ensemble strategy mitigates weaknesses of individual modules, achieving high precision and coverage.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/DucCuong12/NLP_MAP.git
cd NLP_MAP
