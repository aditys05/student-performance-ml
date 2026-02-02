# Student Performance Prediction (ML Project)

This is a simple Machine Learning project built while learning TensorFlow.

The model predicts a student’s final score based on:
- hours studied
- attendance
- previous score

This project helped me understand:
- how ML models are trained
- how to save and reuse a model
- basic project structure
- using TensorFlow with GPU (WSL2)

## How to run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py

## Updates
- Added feature normalization for more stable predictions
- Saved and reused scaler during prediction