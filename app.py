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
        "overview": "Your tomato plant looks healthy. No disease found.",
        "impact": "No damage. Your plant is growing well.",
        "favorable_conditions": "Not applicable.",
        "chemical_treatment": "No spray needed right now.",
        "prevention": [
            "Check your plants every few days for spots or yellow leaves",
            "Water at the base of the plant, not on the leaves",
            "Give proper fertilizer as needed"
        ],
        "severity": "None"
    },
    'Tomato___Late_blight': {
        "overview": "A serious disease that causes dark, wet-looking patches on leaves and can spread very fast.",
        "impact": "Can destroy your entire crop within a few days if not treated quickly.",
        "favorable_conditions": "Happens more in cool, damp weather and after heavy rain or fog.",
        "chemical_treatment": "Spray Mancozeb or Chlorothalonil on the plant. Repeat every 7-10 days. In bad cases, use Metalaxyl + Mancozeb together.",
        "prevention": [
            "Don't water the leaves directly - water the soil around the plant",
            "Leave enough space between plants for air to pass",
            "Remove and burn/bury infected leaves immediately",
            "Don't plant tomato and potato in the same field next season"
        ],
        "severity": "High"
    },
    'Tomato___Early_blight': {
        "overview": "Dark rings appear on older leaves, usually starting from the bottom of the plant.",
        "impact": "Plant becomes weak and gives fewer, smaller tomatoes if not treated.",
        "favorable_conditions": "Common in warm weather with alternating rain and dry spells.",
        "chemical_treatment": "Spray Chlorothalonil or a copper-based spray as soon as you see spots. Repeat every 7-14 days.",
        "prevention": [
            "Put mulch (dry grass/straw) around the base of the plant",
            "Remove infected bottom leaves right away",
            "Don't water the leaves, only the soil",
            "Change what you plant in this field next season"
        ],
        "severity": "Medium"
    },
    'Tomato___Bacterial_spot': {
        "overview": "Small dark spots appear on leaves and fruit, caused by bacteria, not fungus.",
        "impact": "Leaves fall off and fruit quality drops, reducing how much you can sell.",
        "favorable_conditions": "Spreads fast in warm, wet weather, especially with wind and rain.",
        "chemical_treatment": "Use a copper spray mixed with Mancozeb. Regular fungicides will not work since this is bacterial.",
        "prevention": [
            "Only use good quality, disease-free seeds or seedlings",
            "Don't work in the field when plants are wet",
            "Change crops in this field for 2-3 years",
            "Remove and destroy infected plants"
        ],
        "severity": "Medium"
    },
    'Tomato___Leaf_Mold': {
        "overview": "Yellow spots appear on top of leaves, with a green-gray fuzzy layer underneath.",
        "impact": "Mostly affects plants grown in greenhouses or very humid areas; can reduce yield noticeably.",
        "favorable_conditions": "Happens in high humidity with poor air flow, especially in greenhouses.",
        "chemical_treatment": "Spray Chlorothalonil or a copper-based fungicide. Improve air flow around plants to help the spray work better.",
        "prevention": [
            "Open up space around plants for better air flow",
            "Reduce humidity if growing in a greenhouse/covered area",
            "Don't water the leaves directly",
            "Choose disease-resistant tomato seed varieties if available"
        ],
        "severity": "Medium"
    },
    'Tomato___Septoria_leaf_spot': {
        "overview": "Small round spots with dark edges and gray centers appear on leaves.",
        "impact": "Leaves fall off starting from the bottom, which can reduce your harvest by a lot.",
        "favorable_conditions": "Common after rain or heavy dew, in warm humid weather.",
        "chemical_treatment": "Spray Chlorothalonil or Mancozeb every 7-10 days while conditions stay wet/humid.",
        "prevention": [
            "Remove infected lower leaves as soon as you see them",
            "Put mulch around the base of plants",
            "Water the soil, not the leaves",
            "Rotate your crops every 2-3 years"
        ],
        "severity": "Medium"
    },
    'Potato___healthy': {
        "overview": "Your potato plant looks healthy. No disease found.",
        "impact": "No damage. Your plant is growing well.",
        "favorable_conditions": "Not applicable.",
        "chemical_treatment": "No spray needed right now.",
        "prevention": [
            "Check your field regularly for early signs of disease",
            "Keep a regular watering schedule",
            "Use good quality, disease-free seed potatoes next season"
        ],
        "severity": "None"
    },
    'Potato___Late_blight': {
        "overview": "A serious, fast-spreading disease causing dark, wet-looking patches on leaves and stems.",
        "impact": "Can destroy your entire crop within 1-2 weeks. Also rots stored potatoes.",
        "favorable_conditions": "Happens more in cool, damp weather with high humidity.",
        "chemical_treatment": "Spray copper-based fungicide or Metalaxyl + Mancozeb immediately. Repeat every 5-7 days if weather stays wet.",
        "prevention": [
            "Use only certified, disease-free seed potatoes",
            "Don't water the leaves - water the soil instead",
            "Remove and destroy any random/self-grown potato plants nearby",
            "Only dig up potatoes after the leaves have completely dried and died"
        ],
        "severity": "High"
    },
    'Potato___Early_blight': {
        "overview": "Dark rings appear on older leaves first, usually lower on the plant.",
        "impact": "Reduces the size of your potatoes and overall harvest by 15-30% if not treated.",
        "favorable_conditions": "Common in warm weather with dry and wet periods alternating, and when soil lacks nutrients.",
        "chemical_treatment": "Spray Chlorothalonil or Mancozeb as soon as spots appear. Repeat every 7-14 days.",
        "prevention": [
            "Keep soil well-fertilized, especially with nitrogen",
            "Avoid growing potato/tomato in the same field repeatedly",
            "Clear away infected plant leftovers after harvest",
            "Make sure plants get enough water - avoid stress"
        ],
        "severity": "Medium"
    }
}

management_info = {
    "Tomato___healthy": {
        "management_tips": ["Continue regular scouting", "Keep weeds and fallen leaves cleared", "Use balanced fertilizer based on soil needs"],
        "chemical_use": "No pesticide is needed for a healthy plant. If fertilizer is required, apply it in the early morning or late afternoon when the soil is moist, never during the hottest part of the day, and water lightly afterward."
    },
    "Tomato___Late_blight": {
        "management_tips": ["Remove badly infected leaves and fruit", "Improve spacing and airflow", "Do not work among wet plants"],
        "chemical_use": "Spray early in the morning after dew has dried or in the late afternoon, not in strong midday sun. Treat at the first signs or before several wet days. Do not spray before rain; wait until leaves are dry after rain and follow the label interval."
    },
    "Tomato___Early_blight": {
        "management_tips": ["Remove lower infected leaves", "Mulch to reduce soil splash", "Rotate away from tomato and potato crops"],
        "chemical_use": "Apply at the first spots in the early morning after leaves dry or in the late afternoon. Avoid hot sunny hours, strong wind, and rain. If rain occurs soon after spraying, do not immediately repeat; follow the product label for reapplication and harvest waiting time."
    },
    "Tomato___Bacterial_spot": {
        "management_tips": ["Use pathogen-free seed and disease-free transplants", "Avoid sprinkler irrigation and water splash", "Rotate with a non-host crop and remove volunteer plants and debris"],
        "chemical_use": "Copper bactericides provide partial protection, not a cure. Apply at the first sign or before a warm, wet infection period; UC IPM reports 10-14 day repeats when warm, moist conditions persist. Use only the interval and product registered in India, and do not spray before rain or in midday heat."
    },
    "Tomato___Leaf_Mold": {
        "management_tips": ["Reduce greenhouse humidity", "Ventilate early in the day", "Remove infected lower leaves"],
        "chemical_use": "Spray early morning after foliage dries or late afternoon when humidity is lower, with good leaf coverage. Avoid hot sunny periods and rain. Improve ventilation first, and after rain wait until foliage is dry before any label-approved reapplication."
    },
    "Tomato___Septoria_leaf_spot": {
        "management_tips": ["Remove spotted lower leaves", "Use mulch to prevent soil splash", "Keep tools and hands clean between plants"],
        "chemical_use": "Begin at the first spots or before a wet period. Apply to dry foliage in the early morning after dew dries or in the late afternoon. Avoid midday heat and rain; after rain, wait for dry leaves and reapply only according to the label and harvest interval."
    },
    "Potato___healthy": {
        "management_tips": ["Inspect plants weekly", "Keep rows weed-free", "Use certified seed potatoes next season"],
        "chemical_use": "No pesticide is needed for a healthy plant. If fertilizer is required, apply it in the early morning or late afternoon when the soil is moist, avoid the hottest hours, and water lightly afterward."
    },
    "Potato___Late_blight": {
        "management_tips": ["Use certified seed tubers", "Remove cull piles, volunteer potatoes, and infected debris", "Avoid overhead irrigation and maintain airflow so foliage dries daily"],
        "chemical_use": "Use a protectant before or when disease risk begins during cool, wet weather. UC IPM describes 7-10 day protection intervals, with shorter intervals sometimes needed in cool rain; use the Indian label interval. Spray dry foliage in early morning after dew dries or late afternoon, never just before rain."
    },
    "Potato___Early_blight": {
        "management_tips": ["Maintain balanced nutrition and irrigation to reduce plant stress", "Remove potato refuse and infected debris after harvest", "Monitor older leaves and record disease development"],
        "chemical_use": "Fungicide is justified when early disease may cause economic loss. Apply when the first symptoms appear; UC IPM describes continued protection at 7-10 day intervals when justified. Spray dry foliage after dew dries, avoid midday heat and rain, and use only the Indian product label interval and harvest waiting period."
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

    detected_crop = max(crop_mass, key=crop_mass.get)
    if detected_crop != crop_type:
        return jsonify({
            "error": "crop_mismatch",
            "message": f"This photo looks like a {detected_crop} leaf, but you selected {crop_type}. Please upload a clear {crop_type} leaf photo.",
            "detected_crop_guess": detected_crop
        }), 200

    relevant_indices = crop_class_indices[crop_type]
    relevant_probs = {class_names[i]: float(prediction[i]) for i in relevant_indices}

    total = sum(relevant_probs.values())
    relevant_probs_normalized = {k: (v / total) * 100 for k, v in relevant_probs.items()}

    sorted_probs = dict(sorted(relevant_probs_normalized.items(), key=lambda x: x[1], reverse=True))
    predicted_label = list(sorted_probs.keys())[0]
    confidence = list(sorted_probs.values())[0]

    info = disease_info.get(predicted_label, {})
    management = management_info.get(predicted_label, {})
    weather_advice = [
        "Check the leaves again after rain, fog, or heavy dew.",
        "Water at the soil, not over the leaves, and improve airflow between plants.",
        "Avoid spraying just before rain; follow the product label and allow the leaves to dry first."
    ]
    favorable_conditions = info.get("favorable_conditions", "")
    if "cool" in favorable_conditions.lower():
        weather_advice[0] = "Be especially alert during cool, damp weather and after fog or rain."
    elif "warm" in favorable_conditions.lower():
        weather_advice[0] = "Be especially alert during warm, humid weather and after wet periods."

    return jsonify({
        "crop_type": crop_type,
        "disease": predicted_label,
        "confidence": round(confidence, 2),
        "severity": info.get("severity", "Unknown"),
        "overview": info.get("overview", ""),
        "impact": info.get("impact", ""),
        "favorable_conditions": info.get("favorable_conditions", ""),
        "weather_advice": weather_advice,
        "chemical_treatment": info.get("chemical_treatment", ""),
        "management_tips": management.get("management_tips", []),
        "chemical_use": management.get("chemical_use", "Follow the product label and local agricultural guidance."),
        "prevention": info.get("prevention", []),
        "all_probabilities": {k: round(v, 2) for k, v in sorted_probs.items()}
    })

print("Setup complete. About to start server...")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', debug=False, port=port)