from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import os

print("Starting imports and setup...")

app = Flask(__name__)
CORS(app)

# Load TFLite model
MODEL_PATH = os.path.join("model", "agriguard_model.tflite")
print("Loading TFLite model from:", MODEL_PATH)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("Model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)

class_names = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___healthy'
]

treatment_map = {
    'Potato___Early_blight': "Apply fungicide (e.g., chlorothalonil). Remove infected leaves. Rotate crops yearly.",
    'Potato___Late_blight': "Apply copper-based fungicide immediately. Destroy infected plants to prevent spread.",
    'Potato___healthy': "No disease detected. Continue regular monitoring and good field hygiene.",
    'Tomato___Bacterial_spot': "Use copper-based bactericide. Avoid overhead watering. Remove infected leaves.",
    'Tomato___Early_blight': "Apply fungicide. Mulch around base. Avoid wetting leaves while watering.",
    'Tomato___Late_blight': "Apply fungicide immediately. Remove and destroy infected plants.",
    'Tomato___Leaf_Mold': "Improve air circulation. Reduce humidity. Apply fungicide if severe.",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves. Apply fungicide. Avoid overhead irrigation.",
    'Tomato___healthy': "No disease detected. Continue regular monitoring and good field hygiene."
}

crop_class_indices = {
    "Potato": [i for i, name in enumerate(class_names) if name.startswith("Potato")],
    "Tomato": [i for i, name in enumerate(class_names) if name.startswith("Tomato")]
}

def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "AgriGuard AI backend is running"})

@app.route('/app')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    crop_type = request.form.get('crop_type')
    if crop_type not in crop_class_indices:
        return jsonify({"error": "crop_type must be 'Tomato' or 'Potato'"}), 400

    file = request.files['file']
    temp_path = "temp_image.jpg"
    file.save(temp_path)

    img = Image.open(temp_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = predict_tflite(img_array)

    relevant_indices = crop_class_indices[crop_type]
    relevant_probs = {class_names[i]: float(prediction[i]) for i in relevant_indices}

    total = sum(relevant_probs.values())
    relevant_probs_normalized = {k: (v / total) * 100 for k, v in relevant_probs.items()}

    sorted_probs = dict(sorted(relevant_probs_normalized.items(), key=lambda x: x[1], reverse=True))
    predicted_label = list(sorted_probs.keys())[0]
    confidence = list(sorted_probs.values())[0]

    os.remove(temp_path)

    return jsonify({
        "crop_type": crop_type,
        "disease": predicted_label,
        "confidence": round(confidence, 2),
        "treatment": treatment_map[predicted_label],
        "all_probabilities": {k: round(v, 2) for k, v in sorted_probs.items()}
    })

print("Setup complete. About to start server...")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', debug=False, port=port)