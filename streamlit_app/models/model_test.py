import joblib
import numpy as np

model = joblib.load("C:/Users/DELL 👑/Desktop/HNG 13/HNG_Stage8/streamlit_app/models/xgb_meta.joblib")
scaler = joblib.load("C:/Users/DELL 👑/Desktop/HNG 13/HNG_Stage8/streamlit_app/models/meta_scaler.joblib")

sample = np.array([[120, 25, 1, 12, 3, 10]])
sample_s = scaler.transform(sample)
print(model.predict_proba(sample_s))
