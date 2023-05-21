import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

with open('stroke_model.pkl','rb') as model:
    rforest = joblib.load(model)


def main():
    st.title('HUMAN STROKE DIAGNOSIS APP')
    st.write('This is a simple app that predicts whether a person has a stroke or not.')
    gender = st.selectbox('Gender',('Male','Female'))
    age = st.number_input('Age',0,100)
    hypertension = st.selectbox('Hypertension',(0,1))
    heart_disease = st.selectbox('Heart Disease',(0,1))
    avg_glucose_level = st.number_input('Average Glucose Level',0,300)
    smoking_status = st.selectbox('Smoking Status',('formely smoked','never smoked','smokes','Unknown'))

    if st.button('Predict'):
       X = np.array([[gender,age,hypertension,heart_disease,avg_glucose_level,smoking_status]])
       X[:,0] = le.fit_transform(X[:,0])
       X[:,5] = le.fit_transform(X[:,5])
       
       prediction = rforest.predict(X)
       
       if prediction == 0:
           st.success('The patient has NO stroke!!')
       else:
           st.success('Patient has stroke!!')
    
if __name__ == '__main__':
    main()


