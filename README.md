## VLG Recruitment Challenge ‘26 — Report

**Author:** CHAYAN AGGARWAL (25323013) — 8198974049

## Table of Contents
- [Introduction](#introduction)
- [Model Development](#model-development)
	- [Data Preprocessing](#data-preprocessing)
	- [Model Architecture](#model-architecture)
	- [Training](#training)
- [Results & Improvements](#results--improvements)
- [Conclusion & Learning Outcomes](#conclusion--learning-outcomes)

### Introduction
This project aims to develop a model to detect anomalous events in surveillance footage. The task poses unique challenges: accurately identifying irregularities from a video dataset that contains some corrupted samples and requires cleaning. The goal is good generalization so the model can distinguish routine background activity from significant deviations in real environments.

### Model Development

#### Data Preprocessing
- **Sequence Generation ("Clip" strategy):** implemented a sliding-window sequence loader that yields clips of 5 consecutive frames (`SEQ_LEN = 5`) instead of single images.
- **Input tensor shape:** `(Batch, Time, Channel, Height, Width)` → e.g. `(Batch_Size, 5, 1, 128, 128)`.
- **Purpose:** providing temporal context (frames t, t-1, t-2, ...) lets the model learn motion dynamics (velocity, direction).
- **Geometric augmentation:** deterministic max-inversion (`max(original, inverted)`) to reduce sensitivity to lighting flicker and emphasize structural motion.

#### Model Architecture
- **Overall:** a ConvLSTM Autoencoder — combines a 2D CNN spatial encoder, a ConvLSTM spatio-temporal bottleneck, and a 2D transposed-conv decoder.
- **Spatial Encoder (2D CNN):** processes each frame to a feature map (e.g. `64 x 16 x 16`) to capture edges and shapes.
- **Spatio-Temporal Bottleneck (ConvLSTM):**
	- Uses convolutional gates so hidden states keep spatial layout (height × width).
	- Input: sequence of feature maps; outputs a spatio-temporal representation preserving spatial locality.
- **Decoder:** transposed convolutions reconstruct frames from ConvLSTM outputs.

#### Training
- **Objective:** sequence reconstruction — model reconstructs the 5-frame clip (optionally from a noisy input).
- **Loss:** Mean Squared Error (MSE) across the full sequence.
- **Optimizer & settings:** Adam with LR = `1e-4` (reduced for stability with recurrent layers). Batch size reduced (e.g. `8`) to fit GPU memory when training on temporal sequences.
- **Hardware:** trained on Kaggle GPU for experiments.

### Results & Improvements
- **Motion sensitivity:** ConvLSTM learns motion speed; faster-than-normal movement (e.g. running) yields higher reconstruction error and is detected as anomalous.
- **Temporal consistency:** LSTM memory reduces false positives caused by single-frame glitches; anomalies must persist across time to strongly affect score.

### Conclusion & Learning Outcomes
- Implemented a custom ConvLSTM cell in PyTorch from scratch.
- Learned to handle 5D tensors `(B, T, C, H, W)` in a deep-learning pipeline.
- Bridged computer vision (CNNs) and sequence modeling (RNNs) for surveillance anomaly detection.




