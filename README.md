# Diabetic Retinopathy Severity Grading

Streamlit app classifying retinal fundus images into 5 DR severity stages
(No DR → Mild → Moderate → Severe → Proliferative DR), using a fine-tuned
Swin Transformer V2 backbone.

## Project background

This model is the best-performing arm from a broader ablation study comparing
imbalance-handling strategies (class weights, augmentation, GAN-based synthetic
data) and loss formulations (categorical cross-entropy vs. CORAL ordinal
regression) across EfficientNetB3 and Swin V2 backbones, trained on the DDR
dataset. Final result: QWK 0.7585, accuracy 0.77.

## Setup

```bash
pip install -r requirements.txt
```

Place your trained model weights at `models/best_model_swin_gan.pt` before running.

## Run locally

```bash
streamlit run app.py
```

## Folder structure

```
├── app.py                      # Main Streamlit app
├── model.py                    # SwinClassifier architecture
├── requirements.txt
├── models/
│   └── best_model_swin_gan.pt  # Trained weights (not included — add your own)
├── utils/
│   └── preprocessing.py        # Image transform matching training pipeline
├── assets/
│   └── sample_images/          # Optional demo images
└── .streamlit/
    └── config.toml             # Theme config
```

## Disclaimer

Educational/portfolio project only — not a diagnostic medical device.
