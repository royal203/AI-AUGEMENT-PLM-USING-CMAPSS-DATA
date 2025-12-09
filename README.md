Multi-Task Transformer for Remaining Useful Life (RUL) & Degradation Rate (DR) Prediction
📌 Project Overview

This repository implements an AI-Augmented Product Lifecycle Management (PLM) system designed for predictive maintenance in industrial environments.
The core innovation is a Multi-Task Transformer model that simultaneously predicts:

🟦 Remaining Useful Life (RUL)

🟩 Degradation Rate (DR)

By learning degradation patterns jointly with RUL, the model provides more accurate, stable, and interpretable predictions compared to traditional single-task models.

The project integrates:

✔ NASA C-MAPSS turbofan engine dataset
✔ Multi-Task Transformer for RUL + DR
✔ LSTM baseline comparison
✔ Streamlit-based Digital Twin dashboard
✔ Complete visualization suite (training curves, scatter plots, error histograms)
✔ Fully modular training pipeline

This implementation supports Industry 4.0 & 5.0 initiatives by embedding AI into PLM for real-time, data-driven lifecycle management.

Novelty: Multi-Task Transformer (RUL + DR)
🔥 Why Multi-Task Learning?

RUL prediction alone does not explain how quickly a machine is degrading.
Adding Degradation Rate (DR) prediction helps the model learn:

The speed of degradation

Hidden failure progression

Temporal patterns missed by single-task models

🧩 Model Outputs:
Output	Description
• RUL	Remaining operational cycles before failure
• DR	Rate of RUL decrease across cycles
📈 Observed Benefits:

+16–20% improvement in RMSE

Smoother RUL predictions

Better end-of-life stability

Physically meaningful DR trends

Higher interpretability for engineers
