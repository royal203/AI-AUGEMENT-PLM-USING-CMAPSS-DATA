# CMAPSS Digital Twin (RUL prediction)


This repository implements a Transformer-based Remaining Useful Life (RUL) regressor for the NASA CMAPSS dataset and a Streamlit UI.


## Quick start
1. Clone repo
2. Place raw CMAPSS files in `data/raw/` (e.g. `train_FD001.txt`)
3. Create virtualenv and install requirements
4. Run `notebooks/prepare_fd001.ipynb` (or run `python scripts/prepare_data.py`) to generate processed data
5. Train: `python src/train.py --processed data/processed/fd001_processed.joblib`
6. Start UI: `streamlit run app/streamlit_app.py`


## Structure
See the repository layout in the project root.


## Citation
If you use this work, cite the original CMAPSS dataset and the transformer + gated conv paper used in your thesis.
