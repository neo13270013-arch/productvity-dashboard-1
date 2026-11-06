Upload these files together to your Streamlit repository:
- app.py
- productivity_model.joblib
- productivity_metadata.json
- everyday_life_dataset.csv
- requirements.txt

Holdout metrics: R2=0.680, MAE=0.714, RMSE=0.899
Previous R2 in metadata: 0.652

Feature transformations: log1p for steps, exercise_minutes, caffeine_mg and screen_time_hours; bedtime is encoded cyclically; date is excluded from model inputs.
The deadline-risk indicator remains rule-based, not a trained deadline model.
