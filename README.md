# Explainable Multimodal Depression Detection using Behavioural, Speech and Language Analysis on the DAIC-WOZ Dataset

## Project Overview
This is a research-grade capstone project for Multimodal Depression Detection using the DAIC-WOZ dataset.

## Architecture Summary
The project contains three independent AI agents:
1. Visual Behaviour Agent (OpenFace temporal features)
2. Hybrid Audio Agent (WavLM + Prosody)
3. Text Agent (DeBERTa-v3)

Outputs are fused using Cross-Attention, Gated Fusion, and Reliability Learning.

## Folder Explanation
- `configs/`: Configuration files and dataclasses.
- `data/`: Raw and processed dataset files.
- `datasets/`: PyTorch Dataset components.
- `models/`: Agent and fusion architectures.
- `preprocessing/`: Feature extraction for each modality.
- `training/`: Training loops and callbacks.

## Installation
```bash
conda env create -f environment.yml
conda activate capstone_env
```

## Environment Creation
Use `environment.yml` or `requirements.txt`.

## Future Roadmap
- Model Implementations
- Model Fusion
- Explainability Integration

## Citation
*Placeholder for Citation*
