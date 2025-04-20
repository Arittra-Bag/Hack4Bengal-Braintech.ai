from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import SegformerForImageClassification
import google.generativeai as genai
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gemini API
genai.configure(api_key="####")
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Load models
try:
    mri_classifier = tf.keras.models.load_model("models/alzheimers_detection_model.h5")

    alzheimers_model = SegformerForImageClassification.from_pretrained('nvidia/mit-b1')
    alzheimers_model.classifier = torch.nn.Linear(alzheimers_model.classifier.in_features, 4)
    alzheimers_model.load_state_dict(torch.load('models/alzheimers_model.pth', map_location=torch.device('cpu')))
    alzheimers_model.eval()

    brain_tumor_model = SegformerForImageClassification.from_pretrained('nvidia/mit-b1')
    brain_tumor_model.classifier = torch.nn.Linear(brain_tumor_model.classifier.in_features, 4)
    brain_tumor_model.load_state_dict(torch.load('models/brain_tumor_model.pth', map_location=torch.device('cpu')))
    brain_tumor_model.eval()
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Class labels
mri_classes = ["Brain MRI", "Not a Brain MRI"]
alzheimers_classes = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
brain_tumor_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
            
        model_type = request.form.get('model_type')
        if not model_type:
            return jsonify({'error': 'No model type specified'}), 400
            
        if model_type not in ["Alzheimer's", "Brain Tumor"]:
            return jsonify({'error': 'Invalid model type'}), 400

        # Open and validate image
        try:
            image = Image.open(file)
            image = image.convert('RGB')  # Convert to RGB to ensure compatibility
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

        # MRI Validation
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        try:
            mri_prediction = mri_classifier.predict(image_array)
            mri_class = mri_classes[np.argmax(mri_prediction)]
        except Exception as e:
            return jsonify({'error': f'Error during MRI validation: {str(e)}'}), 500

        if mri_class == "Not a Brain MRI":
            return jsonify({'result': "Not a Brain MRI", 'confidence': None, 'report': "The uploaded image does not appear to be a brain MRI scan. Please upload a valid brain MRI scan."})

        # MRI Classification
        try:
            image_tensor = transform(image).unsqueeze(0)
            
            if model_type == "Alzheimer's":
                with torch.no_grad():
                    outputs = alzheimers_model(image_tensor).logits
                predicted_class = alzheimers_classes[torch.argmax(outputs).item()]
                confidence = torch.max(torch.nn.functional.softmax(outputs, dim=1)).item() * 100
            else:  # Brain Tumor
                with torch.no_grad():
                    outputs = brain_tumor_model(image_tensor).logits
                predicted_class = brain_tumor_classes[torch.argmax(outputs).item()]
                confidence = torch.max(torch.nn.functional.softmax(outputs, dim=1)).item() * 100
        except Exception as e:
            return jsonify({'error': f'Error during classification: {str(e)}'}), 500

        # Generate Report
        try:
            report = generate_medical_report(predicted_class)
        except Exception as e:
            report = f"Error generating report: {str(e)}"

        return jsonify({
            'result': predicted_class,
            'confidence': confidence,
            'report': report
        })

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

def generate_medical_report(diagnosis):
    try:
        prompt = f"Generate a detailed medical report for {diagnosis}, including causes, symptoms, treatments, and prognosis. Conclude with: Team BrainTech.ai."
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating report: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
