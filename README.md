Engineered an unsupervised Domain-Adaptive Pre-Training (DAPT) pipeline using Unsloth, PyTorch, and Hugging Face TRL to inject specialized raw domain knowledge into open-weight 7B parameter LLMs.
Formatted and tokenized raw unstructured text corpora into continuous 2048-token packed windows for high-throughput Causal Language Modeling (CLM) on single-GPU hardware.
Implemented automated Perplexity (PPL) evaluation scripts comparing Cross-Entropy Loss on target domain text before vs. after adaptation, achieving significant perplexity reduction and zero token embedding hallucination.
