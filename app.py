from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained CNN
model = tf.keras.models.load_model("balanced_waste_cnn_best.keras")

# Class mapping from Kaggle training
classes = ["hazardous", "organic", "recyclable"]


@app.route("/predict", methods=["POST"])
def predict():
    print("PREDICT REQUEST RECEIVED", flush=True)
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    # Prepare image
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))

    # IMPORTANT:
    # Do NOT divide by 255 here.
    # The CNN already contains Rescaling(1./255).
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    print("STARTING MODEL PREDICTION", flush=True)
    predictions = model.predict(image_array, verbose=0)[0]
    print("MODEL PREDICTION COMPLETED", flush=True)

    index = int(np.argmax(predictions))
    predicted_class = classes[index]
    confidence = float(predictions[index])

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 4)
    })


@app.route("/", methods=["GET"])
def home():
    return "Waste Classification API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)