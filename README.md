![Diabetes](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/diabetes_mainbanner.PNG?raw=true)

# Early Risk Diabetes Predictor Machine Learning and App

## Context

What is Diabetes? Diabetes is a chronic disease that affects how your body turns food into energy. It occurs when your blood glucose, also known as blood sugar, is too high. There are three main types of diabetes: Type 1, Type 2, and gestational diabetes. Each type affects the body in different ways, but they all involve problems with insulin production or use. Diabetes is one of the most important diseases Global Impact: Diabetes affects millions of people worldwide. The World Health Organization (WHO) reports that about 422 million people have diabetes, particularly in low- and middle-income countries. Health Risks: If not managed properly, diabetes can lead to serious complications such as heart disease, stroke, kidney failure, blindness, and lower limb amputation. Rising Numbers: The prevalence of diabetes is steadily increasing, making early detection and management crucial.

## Problem Statement

**Why an Early Diabetes Predictor App?** 
Early detection of diabetes can significantly improve the quality of life and reduce the risk of severe complications. Traditional methods for diagnosing diabetes often involve lab tests, which can be expensive and inconvenient. An app that predicts the risk of diabetes based on readily available symptoms can serve as an initial screening tool, providing an early warning and encouraging people to seek medical advice promptly.

**Dataset source**
The idea for creating this app originated from the availability of a comprehensive dataset from the UCI Machine Learning Repository: the Early Stage Diabetes Risk Prediction Dataset. This dataset includes information collected through questionnaires from patients at the Sylhet Diabetes Hospital in Sylhet, Bangladesh, and it has been approved by medical professionals.

**The dataset comprises 16 features, including 1 numerical and 15 categorical features**
![Dataset feature](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/variable_name.PNG)

 ## Insights
* Model Performance: The Random Forest model provided a reliable and balanced performance, making it a suitable choice for deployment.
* Feature Importance: Based on the model, symptoms such as polyuria (Abnormally large amount of urine production) and polydipsia (excessive thirst, commonly a response to dehydration or high blood sugar), are strong indicators of early diabetes risks.

## Conclusion
 we selected the tuned Random Forest as the best model due to its balanced performance without significantly compromising other parameters. The model has a high recall score on the test data by 98.9% with relatively smaller False Negative and Small False Positive.

## Recommendations
* Health Monitoring: Incorporate regular health check-ups and symptom tracking to identify potential risks early.
* Further Research: Explore additional data sources and refine models to enhance prediction accuracy.
* User Education: Educate users about the importance of early detection and the symptoms of diabetes.

## Assets
* [Diabetes risk WebApps](https://early-risk-diabetes-predictor.streamlit.app/) via streamlit (Please, Feel free to check ^_^)
* [Presentation (PDF)](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/Developing%20Early%20Diabetes%20Risk%20predictor.pdf)
