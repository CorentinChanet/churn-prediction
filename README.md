
<div align='center'>

<img width = "200" src = https://becode.org/app/uploads/2020/03/cropped-becode-logo-seal.png>

</div>

# Churn prediction

This project was part of Becode AI Bootcamp

App deployed on [Streamlit](https://share.streamlit.io/corentinchanet/churn-prediction/main) 
and [Heroku](https://corentin-churn-prediction.herokuapp.com/)

## Table of contents
[Description](#Description)  
[Installation](#Installation)  
[Usage](#Usage)

## Description

This application aims at predicting churning customers from a bank.
The original dataset as well as a description of the project can be 
found on [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers)

A particular feature of this otherwise extremely clean dataset is that it is
highly unbalanced; only 16.07% of customers are flagged as "attrited". 

The streamlit application is divided into 2 sections:
1. **Exploratory Data Visualization**. The goal of this section is to display 
   important features of the dataset, and to be able to visualize data points
   in a 3D scatter plot with the binary label (attrited / existing). <br></br>
2. **Prediction**. At the top of this section one can find KPIs regarding the chosen
model from the sidebar, including the parameters of the trained model. Users
   can also display a more advanced dashboard with single predictions from the
   testing set, and the SHapley Additive exPlanation (SHAP) relative to this
   particular model's decision.


## Installation
1. Clone the repository:
```
git clone https://github.com/CorentinChanet/churn-prediction
``` 
2. Install the required libraries:
```
pip install -r requirements.txt
```

## Usage
To start the program on your local machine:
```
streamlit run streamlit_app.py
```

