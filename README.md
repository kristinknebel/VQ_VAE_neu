# VQ_VAE_neu – PTB-XL Latent Space of Heartbeats

Ziel: PTB-XL-ECGs in einzelne Herzschläge segmentieren, im Latentraum (VQ-VAE) auf einer Kugel visualisieren und pro EKG den Anteil pathologischer Beats (z. B. Kammerflimmern) quantifizieren.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
