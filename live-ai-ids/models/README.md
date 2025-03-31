# Models Directory

This directory stores trained machine learning models for the Live AI-IDS system.

## Contents

- `final_model.pt`: The main trained model used for detection
- `checkpoint_epoch_*.pt`: Checkpoint models saved during training

## Notes

- Models are automatically saved here when running `train_model.py`
- These files are excluded from version control by default (see .gitignore)
- Models contain both the neural network weights and metadata like:
  - Input size
  - Number of classes
  - Class names
  - Scaler parameters
  - Label encoder
