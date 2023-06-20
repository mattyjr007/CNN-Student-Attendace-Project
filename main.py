import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import time
import datetime
# Get the current date
current_date = datetime.date.today()


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
    predictions = pred_probabilities > 0.8
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

data = pd.read_csv('attendance.csv')

with header:
    st.title("CNN: Attendance Capture System")


    student_image = st.camera_input("Input Student Image")
    st.write("")
    name = st.text_input("Please Enter Student Name.")
    msg = st.empty()

    if student_image is not None:
        if name.strip() == "":
            msg.warning("Please enter a name")
        else:
            # To read image file buffer as a PIL Image:
            img = Image.open(student_image)
            matricn = take_attendance(student_image,msg)

            if matricn is not None:
                data = pd.concat([data, pd.DataFrame.from_records([{'Name':name,'Matric_No':matricn,'Time':current_date}])])
                data.drop_duplicates(inplace=True,keep=False)
                data.to_csv("attendance.csv",index=False)

data_container = st.container()

with data_container:
    data = pd.read_csv('attendance.csv')
    data.drop_duplicates(inplace=True)
    data.to_csv('attendance.csv',index=False)

    st.header("Student Attendance")
    st.dataframe(data.astype(str),use_container_width=True)

