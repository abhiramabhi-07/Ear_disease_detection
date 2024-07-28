from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('mymodel.h5')

def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

def predict_class(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    return predicted_class_index

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                # Read image data from request stream
                img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
                # Predict the class
                predicted_class_index = predict_class(img)
                # Map the index to class label
                class_labels = ["Chronic otitis media", "Earwax plug", "Myringosclerosis", "Normal"]
                predicted_class = class_labels[predicted_class_index]
                return render_template('result.html', prediction=predicted_class)
            except Exception as e:
                return f"Error processing image: {e}", 500
    return redirect(url_for('index'))  # Redirect to home if no file is uploaded

if __name__ == '__main__':
    app.run(debug=True)
