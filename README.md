# Early Risk Diabetes Predictor Machine Learning and App

![Diabetes](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/diabetes_risk_predictor.PNG)

## Context

What is Diabetes? Diabetes is a chronic disease that affects how your body turns food into energy. It occurs when your blood glucose, also known as blood sugar, is too high. There are three main types of diabetes: Type 1, Type 2, and gestational diabetes. Each type affects the body in different ways, but they all involve problems with insulin production or use. Diabetes is one of the most important diseases Global Impact: Diabetes affects millions of people worldwide. The World Health Organization (WHO) reports that about 422 million people have diabetes, particularly in low- and middle-income countries. Health Risks: If not managed properly, diabetes can lead to serious complications such as heart disease, stroke, kidney failure, blindness, and lower limb amputation. Rising Numbers: The prevalence of diabetes is steadily increasing, making early detection and management crucial.

## Problem Statement

**Why an Early Diabetes Predictor App?** 

![Why](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/why%20diabetes%20prediction%20app.PNG)

Early detection of diabetes can significantly improve the quality of life and reduce the risk of severe complications. Traditional methods for diagnosing diabetes often involve lab tests, which can be expensive and inconvenient. An app that predicts the risk of diabetes based on readily available symptoms can serve as an initial screening tool, providing an early warning and encouraging people to seek medical advice promptly.

**Dataset source**

The idea for creating this app originated from the availability of a comprehensive dataset from the UCI Machine Learning Repository: the Early Stage Diabetes Risk Prediction Dataset. This dataset includes information collected through questionnaires from patients at the Sylhet Diabetes Hospital in Sylhet, Bangladesh, and it has been approved by medical professionals.

**The dataset comprises 16 features, including 1 numerical and 15 categorical features**
![Dataset feature](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/variable_name.PNG)

**Analytical approach**

* Firstly, we conducted exploratory data analysis of the given dataset to gain some insights.
* Secondly, we developed models based on several algorithms consisting of Random forest, XGBoost, GradientBoosting, CatBoost, LightGBM, Decision Tree, Logistic Regression, AdaBoost, and KNN.
* We compared the performance of Machine learning models based on 'Recall' evaluation.
* The best two models were adjusted further through hyperparameter tunning.
* Then, we selected the best model and evaluated using feature importance and Local Interpretable Model-Agnostic Explanations (LIME).
* We deployed the model in our web app using streamlit.

## Results and Insights

**Model performance before hyperparameter tunning**

![Model before tunning](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/recall_train_test_before_tuning.PNG)

* Random Forest and XGBoost achieved the highest mean recall scores on the test set, indicating their effectiveness in identifying positive cases of diabetes.


**Model performance after parameter tunning**
![Model after tunning](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/Model%20performance%20after%20tunning.PNG)


**Confusion matrix of Random Forest and XGBoost**
![Confusion matrix both](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/confusion_matrix_both_rf_xgb.PNG)

* XGBoost achieved a perfect recall score, indicating it correctly identified all positive cases in the test set. However, it had a higher false positive rate, misclassifying 24 negative cases as positive.
* Random Forest achieved nearly perfect recall with only one false negative while maintaining a lower false positive rate compared to XGBoost.


**Random Forest as The Selected Model for Machine Learning App**

While XGBoost achieved a perfect recall, it came at the cost of a higher false positive rate. This could lead to more individuals without diabetes being flagged, causing unnecessary concern and potential overuse of medical resources. Random Forest, on the other hand, provided a balanced performance with high recall and a lower false positive rate. Thus, Random Forest is the preferred model due to its reliable and balanced performance.

**Learning curve**
![Learning curve](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/Learning%20curve.PNG)


**Feature importance**

![Feature importance](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/feature%20importance.PNG)

* Based on the model, symptoms such as polyuria (Abnormally large amount of urine production) and polydipsia (excessive thirst, commonly a response to dehydration or high blood sugar), are strong indicators of early diabetes risks.

**Model explanation using LIME for a patient who was diagnosed as positive**

![LIME positive](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/Lime%20positive%20diabetes.PNG)

**Model explanation using LIME for patient who was diagnosed as negative**

![LIME Negative](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/Lime%20Negative%20diabetes.PNG)

## Conclusion
we selected the tuned Random Forest as the best model due to its balanced performance without significantly compromising other parameters. The model has a high recall score on the test data by 98.9% with relatively smaller False Negative and Small False Positive.

## Recommendations
* Health Monitoring: Incorporate regular health check-ups and symptom tracking to identify potential risks early.
* Further Research: Explore additional data sources and refine models to enhance prediction accuracy.
* User Education: Educate users about the importance of early detection and the symptoms of diabetes.

## Assets
* [Diabetes risk WebApps](https://early-risk-diabetes-predictor.streamlit.app/) via streamlit (Please, Feel free to check ^_^)
* [My article](https://medium.com/@harishmuh/developing-an-early-diabetes-risk-predictor-app-using-machine-learning-39e246fb337d)
* [Presentation (PDF)](https://github.com/harishmuh/Early-Risk-Diabetes-predictor-Machine-Learning-and-app/blob/main/slide/Developing%20Early%20Diabetes%20Risk%20predictor.pdf)
