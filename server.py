from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# ✅ Load the saved model
model_path = "model/energy_consumption_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    model = None  # Prevent crashes if the model is missing

app = FastAPI()

# ✅ Enable CORS
origins = [
    "http://localhost:3000",  # React (Create React App)
    "http://localhost:5173",  # React (Vite)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define input model with lowercase field names (matching frontend)
class EnergyInput(BaseModel):
    timestamp: str = datetime.now().isoformat()  # ✅ Auto-generate timestamp
    temperature: float
    humidity: float
    squareFootage: float
    occupancy: int
    hvacUsage: str
    lightingUsage: str
    renewableEnergy: float
    dayOfWeek: str
    holiday: str

@app.get("/")
def read_root():
    return {"message": "FastAPI is running and connected to energy-ui!"}

@app.post("/predict")
def predict_energy(data: EnergyInput):
    # ✅ Validate dayOfWeek
    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    if data.dayOfWeek not in day_mapping:
        raise HTTPException(status_code=400, detail="Invalid dayOfWeek value")

    # ✅ Convert categorical inputs
    hvac_on = 1 if data.hvacUsage.lower() == "on" else 0
    lighting_on = 1 if data.lightingUsage.lower() == "on" else 0
    holiday_flag = 1 if data.holiday.lower() == "yes" else 0
    day_encoded = day_mapping[data.dayOfWeek]

    # ✅ Ensure model is available
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train and save your model first.")

    # ✅ Prepare input data
    input_data = np.array([[data.temperature, data.humidity, data.squareFootage, 
                            data.occupancy, hvac_on, lighting_on, data.renewableEnergy,
                            day_encoded, holiday_flag]])

    # ✅ Make prediction
    prediction = model.predict(input_data)

    return {
        "predictedEnergy": prediction[0],
        "timestamp": data.timestamp  # ✅ Return the timestamp
    }

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
