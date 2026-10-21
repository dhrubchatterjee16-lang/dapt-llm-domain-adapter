# DAPT LLM Domain Adapter: Continued Pre-Training for Chemical & Process Systems Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DAPT](https://img.shields.io/badge/Technique-Domain--Adaptive%20Pre--Training-blueviolet)]()
[![Unsloth](https://img.shields.io/badge/Acceleration-Unsloth%202x%20Faster-blue)]()

Production pipeline for Domain-Adaptive Pre-Training (DAPT) specializing open-source foundational LLMs (Llama-3, Mistral, Qwen-2.5) on dense chemical engineering corpora, technical standards (API, ISA, ASME), patent disclosures, and thermodynamic equations of state.

---

## 📚 Methodology

Standard foundation models understand general coding and literature, but suffer high perplexity on specialized chemical equations, IUPAC nomenclature, and process engineering nomenclature.

1. **Curated Domain Corpora:**
   - Pre-processed open chemical patent texts (USPTO/EPO).
   - Chemical engineering peer-reviewed literature (Ind. Eng. Chem. Res., AIChE Journal, Chem. Eng. Sci.).
   - Process safety and equipment design manuals (Perry's Chemical Engineers' Handbook domain segments).
2. **Tokenization Vocabulary Adaptation:**
   - Augments standard BPE tokenizers with sub-word chemical motifs (e.g., `-sulfonyl-`, `trifluoromethyl`, `[OH-]`).
3. **Continual Pre-Training with Packing:**
   - Implements FlashAttention-2 with sequence packing for maximum GPU throughput and zero padding waste.

---

## 🚀 Quickstart

```bash
git clone https://github.com/dhrubchatterjee16-lang/dapt-llm-domain-adapter.git
cd dapt-llm-domain-adapter
pip install -r requirements.txt
python train_dapt.py --model_name "Qwen/Qwen2.5-7B" --dataset_dir "./data/cheme_corpus"
```
