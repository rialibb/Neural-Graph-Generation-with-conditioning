# Neural Graph Generation with Textual Conditioning

This project addresses the problem of **graph generation based on natural language descriptions** of structural properties such as number of nodes, edges, clustering coefficient, and more. We leverage **Variational Graph Autoencoders (VGAE)** combined with **Latent Diffusion Models (LDMs)** to generate graphs that align with specified properties.

---

## Project Highlights

- **Text-to-Graph Generation**: Encode natural language descriptions using BERT (`bert-base-uncased`) to guide graph generation.
- **Hybrid Architecture**:
  - **Encoder**: GIN or GAT-based VGAE encodes input graphs.
  - **Latent Denoising**: Diffusion process (linear/cosine) in latent space to refine structure.
  - **Decoder**: Conditioned to ensure property compliance (via concatenation or FiLM).
- **Loss Metric**: Mean Absolute Error (MAE) between target and generated graph statistics.
- **Customization**: Supports multiple scheduler types and conditioning strategies.

---

## 🧪 Directory Structure

```
├── data/                  # Contains train/val/test datasets
├── logs/                 # Logging different model runs
├── models/               # Saved model weights (autoencoder, bert, denoiser)
├── output/               # CSV outputs of generated graphs
├── autoencoder.py        # VGAE definition with GIN/GAT
├── bert_model.py         # BERT-based encoder for text descriptions
├── denoise_model.py      # Latent denoising model using diffusion
├── extract_feats.py      # Utility to extract features from text
├── main.py               # Main training & inference pipeline
├── utils.py              # Preprocessing, beta schedulers, and evaluation
├── README.md             # This file
├── REPORT.pdf            # Full project report
```

---

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data & Train**
   ```bash
   python main.py --train-autoencoder --train-bert --train-denoiser --encoder-type GAT --aggregation-type FiLM
   ```

3. **Evaluate**
   ```bash
   # Evaluate MAE between generated graphs and target properties
   python main.py --train-autoencoder False --train-denoiser False
   ```

---

## 📈 Results

- **Best MAE**: 0.1799 on the test set using GAT + FiLM + fine-tuned BERT + cosine scheduler.
- **Observations**:
  - GAT encoders offer better structural attention.
  - FiLM conditioning yields more controllable generation than simple concatenation.
  - Cosine scheduler improves sample refinement during late diffusion stages.

---

## Citations

Based on:
- [Neural Graph Generator (Evdaimon et al., 2024)](https://arxiv.org/abs/2403.01535)
- [GraphVAE (Simonovsky and Komodakis, 2018)](https://arxiv.org/abs/1802.03480)
