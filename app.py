from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import traceback

# Import your model here
# For example:
# from tensorflow.keras.models import load_model
# model = load_model('path_to_your_model.h5')

app = Flask(__name__)

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
    # Example preprocessing for a typical CNN model
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to your model's input size
    img_array = np.array(img) / 255.0  # Normalize
    
    # Add batch dimension if needed
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(image_array):
    """
    Make predictions using your loaded model
    Replace this with your actual prediction code
    """
    # Example prediction function
    # predictions = model.predict(image_array)
    # result = interpret_predictions(predictions)
    
    # Placeholder for demonstration
    result = {
        "class": "example_class",
        "confidence": 0.95
    }
    
    return result

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Check if image file is included in request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if user submitted an empty form
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            # Get prediction
            result = predict(processed_image)
            
            # Clean up - remove uploaded file (optional)
            os.remove(filepath)
            
            return jsonify(result)
        else:
            return jsonify({"error": f"File type not allowed. Please upload {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    except Exception as e:
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)