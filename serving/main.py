from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import psutil
import os
import json

app = FastAPI()

# 메모리 확인용
process = psutil.Process(os.getpid())

# TensorFlow 모델 로드
model = tf.keras.models.load_model("./model/tensorflow_python_model.h5")

# 필요한 열 목록
required_columns = [
    "amp_temp_1",
    "amp_temp_2",
    "amp_temp_3",
    "amp_temp_4",
    "cpu_temp",
    "interval",
    "pos_1",
    "pos_2",
    "pos_3",
    "pos_4",
    "signal_pack",
    "signal_pick",
    "speed_1",
    "speed_2",
    "speed_3",
    "speed_4",
    "torque_1",
    "torque_2",
    "torque_3",
    "torque_4",
    "vacuum",
    "signal_pick_count",
    "signal_pack_count",
]

class PredictionData(BaseModel):
    amp_temp_1: float
    amp_temp_2: float
    amp_temp_3: float
    amp_temp_4: float
    cpu_temp: float
    interval: float
    pos_1: float
    pos_2: float
    pos_3: float
    pos_4: float
    signal_pack: int
    signal_pick: int
    speed_1: float
    speed_2: float
    speed_3: float
    speed_4: float
    torque_1: float
    torque_2: float
    torque_3: float
    torque_4: float
    vacuum: int
    signal_pick_count: int
    signal_pack_count: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: PredictionData):
    start_time = time.time()
    
    try:
        input_data = np.array([
            [
                data.amp_temp_1,
                data.amp_temp_2,
                data.amp_temp_3,
                data.amp_temp_4,
                data.cpu_temp,
                data.interval,
                data.pos_1,
                data.pos_2,
                data.pos_3,
                data.pos_4,
                data.signal_pack,
                data.signal_pick,
                data.speed_1,
                data.speed_2,
                data.speed_3,
                data.speed_4,
                data.torque_1,
                data.torque_2,
                data.torque_3,
                data.torque_4,
                data.vacuum,
                data.signal_pick_count,
                data.signal_pack_count,
            ]
        ])
        
        input_df = pd.DataFrame(input_data, columns=required_columns)
        
        prediction = model.predict(input_df)

        end_time = time.time()
        execution_time = end_time - start_time
        
        memory_usage_bytes = process.memory_info().rss
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)

        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Memory Usage: {memory_usage_mb:.2f} MB")

        return {"predictions": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
