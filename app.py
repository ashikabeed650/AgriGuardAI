from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import os

print("Starting imports and setup...")

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("model", "agriguard_model.tflite")
print("Loading TFLite model from:", MODEL_PATH)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("Model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

crop_class_indices = {
    "Potato": [i for i, name in enumerate(class_names) if name.startswith("Potato")],
    "Tomato": [i for i, name in enumerate(class_names) if name.startswith("Tomato")]
}

disease_info = {
    'Tomato___healthy': {
        "overview": "The plant shows no signs of disease or pest damage.",
        "impact": "None - plant is healthy and growing normally.",
        "favorable_conditions": "N/A",
        "chemical_treatment": "No treatment needed.",
        "prevention": [
            "Continue regular monitoring for early signs of disease",
            "Maintain proper watering and spacing",
            "Apply balanced fertilizer as per soil test recommendations"
        ],
        "severity": "None"
    },
    'Tomato___Late_blight': {
        "overview": "A fast-spreading fungal disease caused by Phytophthora infestans, affecting leaves, stems, and fruit.",
        "impact": "Can destroy 50-100% of crop yield within days if untreated; historically responsible for the Irish Potato Famine.",
        "favorable_conditions": "Thrives in cool, wet weather (15-20C) with high humidity (above 90%) and prolonged leaf wetness.",
        "chemical_treatment": "Apply Mancozeb or Chlorothalonil-based fungicides every 7-10 days. For severe cases, use Metalaxyl + Mancozeb combination.",
        "prevention": [
            "Avoid overhead irrigation; use drip irrigation instead",
            "Ensure proper plant spacing for airflow",
            "Remove and destroy infected plant debris immediately",
            "Avoid planting tomato/potato in the same field consecutively"
        ],
        "severity": "High"
    },
    'Tomato___Early_blight': {
        "overview": "A fungal disease caused by Alternaria solani, typically starting on older, lower leaves as dark concentric-ring spots.",
        "impact": "Reduces photosynthesis and yield by 20-50% if left untreated; weakens plant over the season.",
        "favorable_conditions": "Favored by warm temperatures (24-29C), high humidity, and alternating wet-dry periods.",
        "chemical_treatment": "Apply Chlorothalonil or Copper-based fungicides at first sign of symptoms, repeat every 7-14 days.",
        "prevention": [
            "Mulch around the base of plants to prevent soil splash onto leaves",
            "Remove lower infected leaves promptly",
            "Avoid overhead watering",
            "Rotate crops with non-solanaceous plants"
        ],
        "severity": "Medium"
    },
    'Tomato___Bacterial_spot': {
        "overview": "Caused by Xanthomonas bacteria, producing small, dark, water-soaked spots on leaves and fruit.",
        "impact": "Can cause significant defoliation and reduce marketable fruit yield by 10-30%.",
        "favorable_conditions": "Spreads rapidly in warm (24-30C), wet, humid conditions, especially with wind-driven rain.",
        "chemical_treatment": "Use Copper-based bactericides combined with Mancozeb; bacterial diseases don't respond to standard fungicides.",
        "prevention": [
            "Use disease-free certified seeds/seedlings",
            "Avoid working in fields when plants are wet",
            "Practice crop rotation (2-3 years)",
            "Remove and destroy infected plant material"
        ],
        "severity": "Medium"
    },
    'Tomato___Leaf_Mold': {
        "overview": "A fungal disease (Passalora fulva) causing yellow spots on upper leaf surfaces and olive-green mold underneath.",
        "impact": "Mainly affects greenhouse-grown tomatoes; can reduce yield by 30-50% in high-humidity environments.",
        "favorable_conditions": "Thrives in high humidity (above 85%) and poor air circulation, common in greenhouses.",
        "chemical_treatment": "Apply Chlorothalonil or Copper-based fungicides; improve ventilation to reduce fungicide dependency.",
        "prevention": [
            "Improve greenhouse/field ventilation",
            "Reduce humidity through proper spacing",
            "Avoid overhead watering",
            "Use resistant tomato varieties where available"
        ],
        "severity": "Medium"
    },
    'Tomato___Septoria_leaf_spot': {
        "overview": "A fungal disease caused by Septoria lycopersici, producing small circular spots with dark borders and gray centers.",
        "impact": "Causes progressive defoliation starting from lower leaves, reducing yield and fruit quality by up to 40%.",
        "favorable_conditions": "Favored by warm, wet, humid weather, especially after rain or heavy dew.",
        "chemical_treatment": "Apply Chlorothalonil or Mancozeb-based fungicides every 7-10 days during favorable conditions.",
        "prevention": [
            "Remove and destroy infected lower leaves",
            "Mulch to prevent soil splash",
            "Avoid overhead irrigation",
            "Practice 2-3 year crop rotation"
        ],
        "severity": "Medium"
    },
    'Potato___healthy': {
        "overview": "The plant shows no signs of disease or pest damage.",
        "impact": "None - plant is healthy and growing normally.",
        "favorable_conditions": "N/A",
        "chemical_treatment": "No treatment needed.",
        "prevention": [
            "Continue regular field monitoring",
            "Maintain proper irrigation schedule",
            "Use certified disease-free seed potatoes"
        ],
        "severity": "None"
    },
    'Potato___Late_blight': {
        "overview": "A highly destructive fungal disease caused by Phytophthora infestans, affecting leaves, stems, and tubers.",
        "impact": "Can cause total crop loss within 7-14 days under favorable conditions; also causes tuber rot in storage.",
        "favorable_conditions": "Cool, wet weather (15-20C) with high humidity and extended leaf wetness periods.",
        "chemical_treatment": "Apply copper-based fungicides or Metalaxyl + Mancozeb immediately at first sign of infection; repeat every 5-7 days in high-risk periods.",
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Avoid overhead irrigation",
            "Destroy volunteer potato plants and infected debris",
            "Harvest tubers only after foliage has fully died back"
        ],
        "severity": "High"
    },
    'Potato___Early_blight': {
        "overview": "A fungal disease caused by Alternaria solani, producing dark concentric-ring spots on older leaves first.",
        "impact": "Reduces photosynthesis and tuber size; yield loss typically 15-30% if untreated.",
        "favorable_conditions": "Warm temperatures (24-29C) with alternating wet and dry periods, and plant stress (e.g., low nitrogen).",
        "chemical_treatment": "Apply Chlorothalonil or Mancozeb-based fungicides at first symptoms, repeat every 7-14 days.",
        "prevention": [
            "Maintain adequate soil fertility, especially nitrogen",
            "Practice crop rotation (avoid solanaceous crops for 2-3 years)",
            "Remove infected plant debris after harvest",
            "Avoid water stress on plants"
        ],
        "severity": "Medium"
    }
}

def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

def looks_like_leaf(img_array):
    img = img_array[0] * 255.0
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    exg = 2 * g - r - b
    green_pixel_ratio = np.mean(exg > 15)
    return green_pixel_ratio > 0.12

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'home.html')

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

    os.remove(temp_path)

    if not looks_like_leaf(img_array):
        return jsonify({
            "error": "not_a_leaf",
            "message": "This doesn't look like a plant leaf photo. Please upload a clear photo of a tomato or potato leaf."
        }), 200

    prediction = predict_tflite(img_array)

    crop_mass = {}
    for crop, indices in crop_class_indices.items():
        crop_mass[crop] = float(sum(prediction[i] for i in indices))

    selected_mass = crop_mass[crop_type]
    other_crop = [c for c in crop_class_indices if c != crop_type][0]
    other_mass = crop_mass[other_crop]

    if other_mass > selected_mass:
        return jsonify({
            "error": "crop_mismatch",
            "message": f"This photo looks more like {other_crop}, not {crop_type}. Please select the correct crop or upload a {crop_type} leaf photo.",
            "detected_crop_guess": other_crop
        }), 200

    relevant_indices = crop_class_indices[crop_type]
    relevant_probs = {class_names[i]: float(prediction[i]) for i in relevant_indices}

    total = sum(relevant_probs.values())
    relevant_probs_normalized = {k: (v / total) * 100 for k, v in relevant_probs.items()}

    sorted_probs = dict(sorted(relevant_probs_normalized.items(), key=lambda x: x[1], reverse=True))
    predicted_label = list(sorted_probs.keys())[0]
    confidence = list(sorted_probs.values())[0]

    info = disease_info.get(predicted_label, {})

    return jsonify({
        "crop_type": crop_type,
        "disease": predicted_label,
        "confidence": round(confidence, 2),
        "severity": info.get("severity", "Unknown"),
        "overview": info.get("overview", ""),
        "impact": info.get("impact", ""),
        "favorable_conditions": info.get("favorable_conditions", ""),
        "chemical_treatment": info.get("chemical_treatment", ""),
        "prevention": info.get("prevention", []),
        "all_probabilities": {k: round(v, 2) for k, v in sorted_probs.items()}
    })

print("Setup complete. About to start server...")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', debug=False, port=port)