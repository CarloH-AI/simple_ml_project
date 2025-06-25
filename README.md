# Simple ML Project – Predicting House-Sale Prices 🏠📈

A lightweight, end-to-end regression pipeline that shows how to turn the well-known **Ames Housing** dataset into a production-ready model.  
The single notebook walks through data download, cleaning, feature engineering and model training in fewer than 200 lines of code. :contentReference[oaicite:0]{index=0}

---

## What’s inside?

| File | Description |
|------|-------------|
| `simple_ml_project.ipynb` | step-by-step notebook (EDA → preprocessing → modelling → evaluation) |
| `dataset.csv` *auto-downloaded* | 2 930 rows × 75 raw columns; one-hot encoding expands this to ~281 features :contentReference[oaicite:1]{index=1} |

---

## Workflow at a glance

1. **Data loading** – The notebook pulls the CSV from Google Drive via `gdown` (you don’t need an account). :contentReference[oaicite:2]{index=2}  
2. **Exploratory analysis** – Quick preview, missing-value audit and a full correlation heat-map. :contentReference[oaicite:3]{index=3}  
3. **Pre-processing** –  
   * one-hot encode categoricals  
   * scale numeric features with `StandardScaler`  
   * drop the target (`Sale_Price`) into `y`  
4. **Modelling** – Baseline **K-Nearest-Neighbours Regressor** tuned over `n_neighbors` and `weights`; best model re-trained on the full training set.  
5. **Evaluation** – MAE / RMSE and residual plots (log–log) highlight under-prediction on the luxury-price tail.  
6. **Next steps** – Notebook suggests trying tree ensembles (XGBoost, Random Forest) or target-transform tricks to tackle skew.

---

## Quick start

```bash
git clone https://github.com/CarloH-AI/simple_ml_project.git
cd simple_ml_project

# (optional) tidy environment
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn gdown jupyter

# fire up the notebook
jupyter lab simple_ml_project.ipynb
