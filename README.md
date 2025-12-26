# InstruNet AI

CNN-Based Music Instrument Recognition System.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train the model (using dummy data for now):
```bash
python src/train.py
```
This will save the model to `models/instrunet_model.h5`.

### Running the App
To start the Streamlit dashboard:
```bash
streamlit run src/app.py
```

## Project Structure
- `data/`: Data directory
- `src/`: Source code
  - `preprocessing.py`: Audio processing utilities
  - `model.py`: CNN model definition
  - `train.py`: Training script
  - `app.py`: Streamlit application
- `models/`: Saved models
- `notebooks/`: Jupyter notebooks
