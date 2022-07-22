# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:43:37 2022

@author: Abirami
"""


import requests
import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model

app=Flask(__name__)

model1=load_model(r'C:\Users\Abinaya Venkatesh\Desktop\Kamalaveni\AI-Day-5-Flask-DL-Deployment\Flask\fruit pred.h5')
model=load_model(r'C:\Users\Abinaya Venkatesh\Desktop\Kamalaveni\AI-Day-5-Flask-DL-Deployment\Flask\veg pred.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname('__file__')
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        print(plant)
        if(plant=="vegetable"):
            pred=np.argmax(model.predict(x),axis=1)
            print(pred[0])
            index=['Oopps!! Your pepper plant is infected by Bacterial Leaft Spot. The disease cycle can be stopped by using the Sango formula for disinfectants. Bleach treatment and hot water treatment is also helpful.', 'Yaayy!! Your pepper plant is healthy. But, take the necessary precautions like, putting the plant where it gets at least 10 hours of direct sunlight. Keep soil evenly moist for good growth. Peppers need well draining soil that is rich and loamy, but avoid too much nitrogen in the soil. Too much nitrogen can cause plenty of leaves and little to no peppers. Your soil should have a pH between 6.0 and 6.5.', 'Oopps!! Your potato plant is Early Blight. Avoid irrigation in cool cloudy weather and time irrigation to allow plants time to dry before nightfall. Protectant fungicides (e.g. maneb, mancozeb, chlorothalonil, and triphenyl tin hydroxide) are effective.', 'Oopps!! Your potato plant is Late Blight. The late blight can be effectively managed with prophylactic spray of mancozeb, cymoxanil+mancozeb or dimethomorph+mancozeb.', 'Yaayy!! Your potato plant is healthy. But, take the necessary precautions like, putting the plant where it gets at least 10 hours of direct sunlight. Potatoes do best in well-drained and fertile soil. Maintain the pH between 5.0 and 5.5. Keep soil evenly moist for good growth. Do not add large amounts of organic matter to the soil as it may contribute to potato scab, a disease that frequently infects potatoes.', ' Oopps!! Your tomato plant is effected by bacterial spots. To protect the uninfected plants remove the infected leaves and bury or burn them s there is no cure for this infection. To prevent future infections plant pathogen-free seeds or transplants to prevent the introduction of bacterial spot pathogens on contaminated seed or seedlings.', 'Oopps!! Your tomato plant is late blight. Early treatment for this disease is needed. Fungicides like e Daconil fungicides from GardenTech brand prevent, stop, and control late blight and more than 65 types of fungal disease. Planting resistant cultivars and watering the plants early in the mornings help to prevent this infection.', 'Oopps!! Your tomato plant has leaf molds. Watering the plants early in the mornings help them to get sufficient time to dry out. Fungicidal sprays mostly calcium chloride based sparys help in getting rid of leaf molds.', 'Oopps!! Your tomato plant is infected by Septoria leaf spot. Removing the infected leaves immediately will curb the spread of infection. Organic and chemical fungicides with chlorothalonil are effective in treatment.']
            text=str(index[pred[0]])
            print(text)
        else:
            pred=np.argmax(model1.predict(x),axis=1)
            print(pred[0])
            index=['Oopps!! Your apple plant is infected by Black Rots. This infection is a fungal infection. To control balck rot, remove the cankers by pruning at least 15 inches below the end and burn or bury them. Treating the sites with the antibiotic streptomycin or a copper-based fungicide will be helpful.', 'Yaayy!! Your apple plant is healthy. But, maintain the soil pH of 6.0 to 7.0 for healthy growth. Avoid planting apples in a low spot where cold air or frost can settle.', 'Oopps!! Your corn plant is infected by Northern Leaf Blight. The primary management strategy to reduce the incidence and severity of NCLB is planting resistant products. Using fungicides is also helpful.', 'Yaayy!! Your corn plant is healthy. But, maintain the soil consistently moist, but not soggy and only need fertilizer every 6 months. It prefers temperatures of 75 to 80 degrees F.', 'Oopps!! Your peach plant is infected by Bacterial Spots. This is a difficult disease to control when environmental conditions favor pathogen spread. Compounds for the treatment include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control.', 'Yaayy!! Your peach plant is healthy. But, you should have deep sandy soil that ranges from a loam to a clay loam for healthy growth. Poor drainage in the soil will kill the root system of growing peach trees, so make sure the soil is well drained. Growing peach trees prefer a soil pH of around 6.5.']
            text=str(index[pred[0]])
            print(text)
        return text
        
 
if __name__=='__main__':
    app.run(debug=False)