import gradio as gr
import numpy as np
from keras.models import load_model
import cv2

model_name = "masked_gender_classification_model_pallav1200imgs8lr.h5"
model = model = load_model(model_name)

def predict(img):
    # img = cv2.imread(rf'samples/{img_name}')

    imgp = cv2.resize(img, (224, 224))
    imgp = np.array(imgp)/255.0
    imgp = imgp.reshape(1,224,224,3)
    prediction = model.predict(imgp)[0][0]
    # print(prediction)
    if prediction>0.5:
        text='FEMALE'
    else:
        text='MALE'
    
    img = cv2.putText(img, text, (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
    return img, str(max(prediction, 1-prediction)*100)+"% "+text
    

def greet(img):
    return predict(img)

demo = gr.Interface(
    fn=greet,
    inputs= gr.Image(),
    outputs=["image", "text"]
)

demo.launch()
