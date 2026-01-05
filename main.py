from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import gc

MODEL_PATH = "models/RandomForest.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = None
preprocessor = None


def load_artifacts():
    global model, preprocessor
    if model is None or preprocessor is None:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        gc.collect()


app = FastAPI(title="Cardiovascular Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    Exercise: str
    Heart_Disease: str
    Skin_Cancer: str
    Other_Cancer: str
    Depression: str
    Arthritis: str
    Sex: str

    Height_cm: float
    Weight_kg: float
    BMI: float

    Smoking_History: str
    Alcohol_Consumption: float
    Fruit_Consumption: float
    Green_Vegetables_Consumption: float
    FriedPotato_Consumption: float

    General_Health_Excellent: bool = False
    General_Health_Fair: bool = False
    General_Health_Good: bool = False
    General_Health_Poor: bool = False
    General_Health_Very_Good: bool = False

    Checkup_5_or_more_years_ago: bool = False
    Checkup_Never: bool = False
    Checkup_Within_the_past_2_years: bool = False
    Checkup_Within_the_past_5_years: bool = False
    Checkup_Within_the_past_year: bool = False

    Diabetes_No: bool = False
    Diabetes_Pre: bool = False
    Diabetes_Yes: bool = False
    Diabetes_Pregnancy: bool = False

    Age_Category_18_24: bool = False
    Age_Category_25_29: bool = False
    Age_Category_30_34: bool = False
    Age_Category_35_39: bool = False
    Age_Category_40_44: bool = False
    Age_Category_45_49: bool = False
    Age_Category_50_54: bool = False
    Age_Category_55_59: bool = False
    Age_Category_60_64: bool = False
    Age_Category_65_69: bool = False
    Age_Category_70_74: bool = False
    Age_Category_75_79: bool = False
    Age_Category_80_plus: bool = False


@app.get("/")
def home():
    return {"message": "CVD Risk Prediction API is running"}


@app.post("/predict")
def predict(data: PatientInput):
    load_artifacts()

    df = pd.DataFrame([data.model_dump()])

    df.rename(
        columns={
            "Height_cm": "Height_(cm)",
            "Weight_kg": "Weight_(kg)",
            "Age_Category_80_plus": "Age_Category_80+",
        },
        inplace=True,
    )

    for col, le in preprocessor["label_encoders"].items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    scaler_cols = list(preprocessor["scaler"].feature_names_in_)
    df[scaler_cols] = preprocessor["scaler"].transform(df[scaler_cols])

    if "minmax" in preprocessor and "minmax_columns" in preprocessor:
        df[preprocessor["minmax_columns"]] = preprocessor["minmax"].transform(
            df[preprocessor["minmax_columns"]]
        )

    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prob = float(model.predict_proba(df)[0][1])

    if prob >= 0.20:
        result = "High Risk"
        label = 1
    elif prob >= 0.10:
        result = "Moderate Risk"
        label = 1
    else:
        result = "Low Risk"
        label = 0

    return {
        "heart_disease_prediction": label,
        "risk_probability": round(prob, 3),
        "result": result,
    }
