from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from flask_cors import CORS  # Import CORS

# Load the model
MODEL_PATH = "poultry_disease_model.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)  
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess the image according to your model's requirements
    Modify this function based on your model's preprocessing needs
    """
    return load_and_preprocess_image(image_path)

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image_array):
    """
    Predict poultry disease from image using the trained model
    """
    if model is None:
        return {"error": "Model not loaded"}

    categories = ['coccidiosis', 'healthy', 'ncd', 'salmonella']
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = categories[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    # Return prediction results
    return {
        "class": predicted_class,
        "confidence": confidence,
        "detailed_results": {categories[i]: float(predictions[0][i] * 100) for i in range(len(categories))}
    }

@app.route('/predict', methods=['POST'])
def predict_api():
    print(request)  # Log the entire request object
    try:
        # Check if image file is included in request
        
        
        file = request.files.get('file')
        
        # Log file details
        print(f"Received file: {file.filename}")
        
        # Check if user submitted an empty form
        if file.filename == '':
            print("No selected file")  # Log empty filename
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {filepath}")  # Log file save path
            file.save(filepath)
            
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            # Get prediction
            result = predict(processed_image)
            
            # Clean up - remove uploaded file (optional)
            os.remove(filepath)
            print(f"File {filepath} removed after processing")  # Log file removal
            
            return jsonify(result)
        else:
            print(f"File type not allowed: {file.filename}")  # Log invalid file type
            return jsonify({"error": f"File type not allowed. Please upload {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    except Exception as e:
        print(f"Error occurred: {e}")  # Log exception details
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# Alternative method using in-memory processing (no file saving)
@app.route('/predict-memory', methods=['POST'])
def predict_memory_api():
    try:
        # Check if image file is included in request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if user submitted an empty form
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            # Process directly in memory
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((224, 224))  # Adjust size as needed
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get prediction
            result = predict(img_array)
            
            return jsonify(result)
        else:
            return jsonify({"error": f"File type not allowed. Please upload {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/test-predict', methods=['GET'])
def test_predict_api():
    """
    Test the prediction API using a predefined test image
    """
    try:
        # Path to the test image (use absolute path)
        test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        
        # Check if the test image exists
        if not os.path.exists(test_image_path):
            return jsonify({"error": f"Test image not found at {test_image_path}"}), 404
        
        # Preprocess the image
        processed_image = preprocess_image(test_image_path)
        
        # Get prediction
        result = predict(processed_image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)
   
