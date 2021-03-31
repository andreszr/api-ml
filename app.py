import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
from typing import List
from pydantic import BaseModel

class ItemList(BaseModel):
    data_to_predict: List[float]

# Model
loaded_model = pickle.load(open('./models/model.pkl', 'rb'))

# inicializar app
app = FastAPI()

# ML application
@app.post('/predict/')
async def predict(data:ItemList):
    result = loaded_model.predict(np.array(data.data_to_predict).reshape(1,4))
    return result[0]

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)