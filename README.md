# IBB Assignment 3 - Training CNN for Ear Identification
- Author: Iztok Ramovš

## Instructions for running the project
```bash
# Create virtual environment
python -m venv ./venv
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt

# Preprocessing
python preprocess.py

# Train model and save it to ./pretrained/
python train.py

# Evaluate models
python evaluate.py

```
