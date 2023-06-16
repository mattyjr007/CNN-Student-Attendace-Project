import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

st.set_page_config(page_title="Attendace App", page_icon="😍")


def convertImgToArray(img):
    myImage =img
    TheIMG = image.load_img(myImage,target_size=(256,256))
    newImg = image.img_to_array(TheIMG)
    newImg = np.expand_dims(newImg,axis=0)
    newImg = newImg/255
    return newImg

def take_attendance(img,msg):
    classes =  pickle.load(open("classes.p", "rb"))

    newImg = convertImgToArray(img)

    myModel = load_model('face_recognition_model.keras', compile=False)
    pred_probabilities = myModel.predict(newImg)

    print(pred_probabilities)
    predictions = pred_probabilities > 0.7
    print(predictions)
    prediction = [i for i, x in enumerate(predictions[0]) if x]
    #print(prediction)
    if prediction:
        matric = classes[prediction[0]]
        msg.success(f"Attendance for {matric} Taken Successfully!!")
        return matric
    else:
        msg.error("Student matriculation number not found in database!!")
        return None


header = st.container()


with header:
    st.title("CNN: Attendance Capture System")


    student_image = st.camera_input("Input Student Image")
    msg = st.empty()

    if student_image is not None:
        # To read image file buffer as a PIL Image:
        img = Image.open(student_image)
        matricn = take_attendance(student_image,msg)
