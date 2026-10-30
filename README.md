# LM-JEPA for Symbolic Regression (GSoC 2026 test task, ML4SCI / SYMBA)

A single notebook, [`LM_JEPA_Symbolic_Regression_GSoC2026.ipynb`](LM_JEPA_Symbolic_Regression_GSoC2026.ipynb),
written for the ML4SCI SYMBA GSoC 2026 evaluation tasks:

- **Common Task 1.1:** preprocess and tokenize the AI-Feynman equations (`FeynmanEquations.csv`).
  Formulas are parsed with SymPy and converted to prefix (Polish) notation; vocabulary in `feynman_vocab.json`.
- **Specific Task 2.7:** train a small Transformer with an LLM-JEPA style objective
  (next-token loss + an embedding-space loss between two "views" of each equation: original
  variable names vs. standardized names `s_1, s_2, ...`), and compare with a plain Transformer LM.

## Setup used in the committed run

Pre-LN Transformer, d_model 128, 4 layers, 4 heads; JEPA predictor MLP; λ_JEPA = 0.5;
50 epochs, LR 3e-4. 100 equations split 80 / 10 / 10 (train / val / test). One run, one seed.

## Results (from the notebook's saved outputs, 10 test equations)

| Metric | LM-JEPA | Baseline (no JEPA) |
|---|---|---|
| Token accuracy | 0.4486 | 0.4216 |
| Sequence (exact) accuracy | 0.0000 | 0.0000 |
| Perplexity | 15.47 | 15.43 |
| BLEU-4 | 0.0000 | 0.0000 |

Honest reading: with 10 test equations and a single seed, the token-accuracy gap is not
evidence that JEPA helps, and neither model recovers any equation exactly. This is a
working pipeline, not a result.

## Files

| File | What |
|---|---|
| `LM_JEPA_Symbolic_Regression_GSoC2026.ipynb` | Full pipeline with outputs |
| `FeynmanEquations.csv` | AI-Feynman equation table (Udrescu & Tegmark, 2019) |
| `feynman_vocab.json` | Token vocabulary |
| `lm_jepa_symbolic_regression.pt` | Saved checkpoint (PyTorch pickle; load only if you trust it) |
| `*.png` | Tokenization stats, training curves, embedding PCA |

## Run

Open the notebook in Jupyter or Colab and run all cells (it installs its own dependencies:
torch, pandas, numpy, matplotlib, seaborn, scikit-learn, sympy, tqdm). A GPU is optional;
the model is small.

## Next steps

- TODO(Keyush): ablation λ_JEPA = 0 vs > 0 over ≥ 3 seeds, with symbolic-equivalence accuracy (SymPy `simplify(a - b) == 0`).
- Larger synthetic pre-training set; EMA target encoder; conditioning on numeric data points.
- Compare against a standard symbolic-regression baseline (e.g. PySR).

## References

- Udrescu & Tegmark (2019), *AI Feynman: a Physics-Inspired Method for Symbolic Regression*, arXiv:1905.11481
- Huang, LeCun & Balestriero (2025), *LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures*, arXiv:2509.14252

TODO(Keyush): add a license.
