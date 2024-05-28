import numpy as np
import pandas as pd
import streamlit as st
import pickle
import time
from PIL import Image


actual_patient_data = pd.read_csv('diabetes_data.csv')

converted_data=pd.get_dummies(actual_patient_data, prefix=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'class'], drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(converted_data.drop('class_Positive', axis=1),converted_data['class_Positive'], test_size=0.3, random_state=0)
   
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_train, y_train)

def predict_note_authentication(age,gender,polyuria,polydipsia,weight,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability, delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity):

    prediction=RF_classifier.predict(sc.transform(np.array([[int(age),int(gender),int(polyuria),int(polydipsia),int(weight),int(weakness),int(polyphagia),int(genital_thrush),int(visual_blurring),int(itching),int(irritability), int(delayed_healing),int(partial_paresis),int(muscle_stiffness),int(alopecia),int(obesity)]])))
    print(prediction)
    return prediction

# Set page title and layout
st.set_page_config(page_title="Diabetes risk predictor apps", layout="centered")

# Introduction
st.write("""
        # Diabetes risk predictor apps
        """)



st.write("""
        ## This apps predicts early symptoms of diabetes melitus
        
        The dataset for this prediction was obtained from [Diabetes symptoms dataset](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) by UC Irvine ML repository. 
        """)

# Diabetes Illustration
image = Image.open("diabetes_mainbanner.png")
st.image(image, caption="Diabetes Illustration", width=700)

# Other illustrations
diabetes_positive= Image.open('diabetes_positive.PNG')
diabetes_negative =Image.open('diabetes_negative.PNG')

def main():
    
    st.subheader("Please choose option below based on your conditions and click on 'Predict' to know your status")
    
    age = st.number_input("How old are you?")

    gender = st.radio("Are you male or female?",("Male","Female"))
    if gender == 'Male':
        gender = 1
    else:
        gender = 0
    
    polyuria = st.radio("Do you frequently urinate more than usual?",("Yes","No"))
    if polyuria == 'Yes':
        polyuria = 1
    else:
        polyuria = 0
    link1 = '[What is frequent urination or "polyuria?"](https://en.wikipedia.org/wiki/Polyuria)'
    st.markdown(link1, unsafe_allow_html=True)
       
    polydipsia = st.radio("Are you often feel very thirsty?",("Yes","No"))
    if polydipsia == 'Yes':
        polydipsia = 1
    else:
        polydipsia = 0
    link2 = '[What is excessive thirst or "polydipsia?"](https://en.wikipedia.org/wiki/Polydipsia)'
    st.markdown(link2, unsafe_allow_html=True)

    weight = st.radio("Have you noticed a sudden loss of weight recently?",("Yes","No"))
    if weight == 'Yes':
        weight = 1
    else:
        weight = 0

    weakness = st.radio("Do you often feel weak?",("Yes","No"))
    if weakness == 'Yes':
        weakness = 1
    else:
        weakness = 0

    polyphagia = st.radio("Do you experience extreme hunger?",("Yes","No"))
    if polyphagia == 'Yes':
        polyphagia = 1
    else:
        polyphagia = 0
    link3 = '[What is extreme hunger or "Polyphagia"?](https://en.wikipedia.org/wiki/Polyphagia)'
    st.markdown(link3, unsafe_allow_html=True)
        
    genital_thrush = st.radio("Do you have a yeast infection around the genital area?",("Yes","No"))
    if genital_thrush == 'Yes':
        genital_thrush = 1
    else:
        genital_thrush = 0
    link4 = '[What is a genital yeast infection?](https://www.ticahealth.org/interactive-guide/your-body/genital-problems/genital-thrush/)'
    st.markdown(link4, unsafe_allow_html=True)

    visual_blurring = st.radio("Does your vision sometimes blurry?",("Yes","No"))
    if visual_blurring == 'Yes':
        visual_blurring = 1
    else:
        visual_blurring = 0
    link5 = '[What is blurry vision?](https://en.wikipedia.org/wiki/Blurred_vision)'
    st.markdown(link5, unsafe_allow_html=True)

    itching = st.radio("Do you have itchy skin?",("Yes","No"))
    if itching == 'Yes':
        itching = 1
    else:
        itching = 0
    link6 = '[What is itchy skin?](https://en.wikipedia.org/wiki/Itch)'
    st.markdown(link6, unsafe_allow_html=True) 

    irritability = st.radio("Do you feel irritable often?",("Yes","No"))
    if irritability == 'Yes':
        irritability = 1
    else:
        irritability = 0
    link7 = '[what is Irritability?](https://en.wikipedia.org/wiki/Irritability)'
    st.markdown(link7, unsafe_allow_html=True)

    delayed_healing = st.radio("Do your wounds take longer to heal?",("Yes","No"))
    if delayed_healing == 'Yes':
        delayed_healing = 1
    else:
        delayed_healing = 0

    partial_paresis = st.radio("o you have partial muscle weakness?",("Yes","No"))
    if partial_paresis == 'Yes':
        partial_paresis = 1
    else:
        partial_paresis = 0
    link8 = '[What is partial muscle weakness or "Paresis"](https://en.wikipedia.org/wiki/Paresis)'
    st.markdown(link8, unsafe_allow_html=True)

    muscle_stiffness = st.radio("Do your muscles often feel stiff?",("Yes","No"))
    if muscle_stiffness == 'Yes':
        muscle_stiffness = 1
    else:
        muscle_stiffness = 0

    alopecia = st.radio("Have you experienced hair loss?",("Yes","No"))
    if alopecia == 'Yes':
        alopecia = 1
    else:
        alopecia = 0
    link9 = '[what is hair loss?](https://en.wikipedia.org/wiki/Hair_loss)'
    st.markdown(link9, unsafe_allow_html=True)

    obesity = st.radio("Are you overweight??",("Yes","No"))
    if obesity == 'Yes':
        obesity = 1
    else:
        obesity = 0

    result=""
    if st.button("Predict"):
        result=predict_note_authentication(age,gender,polyuria,polydipsia,weight,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability, delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity)
        if result ==1:
            with st.spinner('Wait for it...'):
                time.sleep(4)
            st.warning('You have some early risk symptoms of Diabetes. Please consult with a Doctor.')
            st.image(diabetes_positive, width=700)
        else:
            with st.spinner('Wait for it...'):
                time.sleep(4)
            st.success("Hurray! You don't have symptoms of Diabetes. Please consult with Doctor for verification.")
            st.image(diabetes_negative, width=700)
    
    

if __name__=='__main__':
    main()


html_temp1 = """
    <div style="background-color:white;padding:10px">
    
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)

