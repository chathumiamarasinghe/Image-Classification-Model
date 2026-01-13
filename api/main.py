# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import os

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model
# MODEL_PATH = os.path.join("..", "saved_models", "1.keras")
# MODEL = tf.keras.models.load_model(MODEL_PATH)

# CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
# IMAGE_SIZE = 256


# @app.get("/ping")
# async def ping():
#     return {"message": "Hello, I am alive"}


# def read_file_as_image(data) -> np.ndarray:
#     image = Image.open(BytesIO(data)).convert("RGB")
#     image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
#     image = np.array(image) / 255.0
#     return image


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image = read_file_as_image(await file.read())
#         img_batch = np.expand_dims(image, 0)

#         predictions = MODEL.predict(img_batch)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]))

#         return {
#             "class": predicted_class,
#             "confidence": confidence
#         }

#     except Exception as e:
#         return {"error": str(e)}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "1.keras")

print("Loading model from:", MODEL_PATH)
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# ⚠️ IMPORTANT: This order MUST match training notebook exactly
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

IMAGE_SIZE = 256


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()

        # 🔍 DEBUG: check image bytes are changing
        print("Received image size:", len(data))

        image = read_file_as_image(data)
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        # 🔍 DEBUG: see raw predictions
        print("Raw predictions:", predictions)

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]))

        print("Predicted class:", predicted_class, "Confidence:", confidence)

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
