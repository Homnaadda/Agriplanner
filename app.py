import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
from CNN import CNN  # Import the CNN class

# Updated path to read the CSV file
disease_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'disease_info.csv'), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'supplement_info.csv'), encoding='cp1252')

# Initialize the CNN model with the correct number of classes
K = 39  # Set this to the number of classes in your dataset
model = CNN(K)  # Use the CNN class from CNN.py
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "trained_model.pth")), strict=False)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(file_path):
    try:
        # Get the uploaded image
        img = Image.open(file_path)
        img = transform(img)

        # Make a prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output)

        return predicted_class.item()

    except Exception as e:
        return {'error': str(e)}  # Ensure this line is properly indented

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')  

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        # Updated line to use absolute path for saving the image
        file_path = os.path.join(os.path.dirname(__file__), 'static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = predict(file_path)  # Call the predict function here
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
