# Data Directory

This directory stores captured network traffic data for training and testing the Live AI-IDS system.

## Contents

- `benign_traffic.csv`: Captured benign network traffic
- `attack_traffic.csv`: Captured attack network traffic
- `combined_traffic.csv`: Combined dataset for training

## Notes

- Data files are automatically generated when using the `--save-data` option with `train_model.py`
- These files are excluded from version control by default (see .gitignore)
