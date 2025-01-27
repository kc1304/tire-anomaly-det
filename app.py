from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from Data_preprocess import preprocess_image
import cv2

# Initialize Flask application
app = Flask(__name__)

# Load the saved model
model = load_model('tire_model.keras')

# Define route for serving index.html
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    file = request.files['image']
    
    # Load the image using PIL
    img = Image.open(file.stream)
    
    # Convert PIL image to NumPy array
    img_array = np.array(img)
    
    # Preprocess the image using OpenCV
    preprocessed_img = preprocess_image(img_array)
    
    # Ensure the image is in 3-channel format after preprocessing
    preprocessed_img_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
    
    # Convert the processed image back to PIL Image
    preprocessed_pil_img = Image.fromarray(preprocessed_img_rgb)
    
    # Resize the image to match the model input shape
    resized_img = preprocessed_pil_img.resize((100, 100))
    
    # Convert image to numpy array and normalize pixel values
    img_array = np.array(resized_img) / 255.0
    
    # Reshape the image array to match model input shape (1, 100, 100, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Since your model outputs a single value, apply thresholding or round off
    predicted_class = int(np.round(prediction[0][0]))  # Binary output: 0 or 1

    # Return the predicted class
    if predicted_class == 1:
        return "The tire is in good condition"
    elif predicted_class == 0:
        return "The tire needs to be replaced"

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
