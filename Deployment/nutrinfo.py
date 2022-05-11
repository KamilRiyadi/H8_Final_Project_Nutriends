from os import access
from re import X
from PIL import Image
from io import BytesIO
import plotly.express as px
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(layout="wide", page_icon="Nutriends_2.png", page_title="Nutriends - Nutrinfo",
                    menu_items={
                                "Get Help": "https://www.linkedin.com/in/kamilriyadi/",
                                "Report a bug": "https://github.com/KamilRiyadi",
                                "About": "### Nutrient Checker App - By Kamil Riyadi"})

# Load Model
model = tf.keras.models.load_model('DenseNet201.h5')

# Load Nutrients Table
DATA_URL = ('Nutrition_Fact.csv')

@st.cache(allow_output_mutation=True)
def load_data(nrows):
    df = pd.read_csv(DATA_URL, nrows=nrows, delimiter=';')
    return df

df = load_data(51) # insert rows count inside the csv files 

# Variable for image
img = None

# Load image function
@st.cache(allow_output_mutation=True)
def load_image_inf(img):
    # x = tf.io.decode_image(img, channels=3)
    x = tf.convert_to_tensor(img)[:,:,:3]
    x = tf.image.resize_with_pad(x, 128, 128)
    return x

# Classes Mapping
Classes = {'apple_pie': 0,
 'bibimbap': 1,
 'caesar_salad': 2,
 'cheesecake': 3,
 'chicken_curry': 4,
 'chicken_wings': 5,
 'chocolate_cake': 6,
 'chocolate_mousse': 7,
 'club_sandwich': 8,
 'creme_brulee': 9,
 'cup_cakes': 10,
 'deviled_eggs': 11,
 'donuts': 12,
 'dumplings': 13,
 'edamame': 14,
 'filet_mignon': 15,
 'french_fries': 16,
 'french_toast': 17,
 'fried_calamari': 18,
 'fried_rice': 19,
 'frozen_yogurt': 20,
 'garlic_bread': 21,
 'greek_salad': 22,
 'grilled_cheese_sandwich': 23,
 'grilled_salmon': 24,
 'hamburger': 25,
 'hot_dog': 26,
 'ice_cream': 27,
 'lasagna': 28,
 'macaroni_and_cheese': 29,
 'macarons': 30,
 'miso_soup': 31,
 'mussels': 32,
 'omelette': 33,
 'onion_rings': 34,
 'oysters': 35,
 'pancakes': 36,
 'panna_cotta': 37,
 'peking_duck': 38,
 'pizza': 39,
 'ramen': 40,
 'sashimi': 41,
 'spaghetti_bolognese': 42,
 'spaghetti_carbonara': 43,
 'steak': 44,
 'strawberry_shortcake': 45,
 'sushi': 46,
 'takoyaki': 47,
 'tiramisu': 48,
 'waffles': 49}

 # Predict function
def predict_image(img):
    inf = load_image_inf(img) 
    res_prob = model.predict(x=tf.expand_dims(inf, axis=0))

    if res_prob.max() > 0.7:

        res = np.argmax(res_prob,axis=1)
        res = res.item()
        inv_map = {v: k for k, v in Classes.items()}
        result = inv_map[res]
        
        col1, col2 = st.columns(2)

        # Food Prediction Section
        with col1:
            title = f"<h2 style='text-align:left'>Food Name : {result}</h2>"
            st.markdown(title, unsafe_allow_html=True)
            st.image(img, use_column_width='auto')

        # Nutritional Info Section
        with col2:
            title2 = f"<h2 style='text-align:left'>Nutritional Facts</h2>"
            st.markdown(title2, unsafe_allow_html=True)
            summary = df[df.index==res]
            summary_ = summary.style.set_precision(2).hide_index()
            serv_size = summary['Serving_Size_(g)'].values[0]
            food_cal = summary['Calories'].values[0]
            st.table(data=summary_)

            data = {'Nutrient':['Total Fat (g)', 'Total Carbs (g)', 'Total Protein (g)'],
                    'Value':[summary['Total_Fat_(g)'].values[0], summary['Total_Carbohydrate_(g)'].values[0], summary['Protein_(g)'].values[0]]
                    }

            # Pie Chart Visualization
            title2 = f"<h2 style='text-align:left'>Calories Source</h2>"
            st.markdown(title2, unsafe_allow_html=True)

            title0 = f"<em style='text-align:left'>*Based on serving size of {serv_size} gram</em>"
            st.markdown(title0, unsafe_allow_html=True)

            graph = pd.DataFrame(data)
            
            fig = px.pie(graph, values='Value', names='Nutrient')
            st.plotly_chart(fig, use_container_width=True)

        # Activity Information
        with st.expander(f'How long it takes to burn {food_cal} calories?'):

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Slow Walking (2 mph) for:", f'{round(food_cal/2.93, 2)} minutes')
            col2.metric("Running (5 mph) for:", f'{round(food_cal/9.38, 2)} minutes')
            col3.metric("Cycling (10-11.9 mph) for:", f'{round(food_cal/7.03, 2)} minutes')
            col4.metric("Light Calisthenic for:", f'{round(food_cal/4.10, 2)} minutes')

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Playing Soccer (non game) for:", f'{round(food_cal/8.22, 2)} minutes')
            col2.metric("Playing Basketball (non game) for:", f'{round(food_cal/7.03, 2)} minutes')
            col3.metric("Boxing (Sparring) for:", f'{round(food_cal/10.55, 2)} minutes')
            col4.metric("Swimming (Leisurely) for:", f'{round(food_cal/7.03, 2)} minutes')
            st.write('*Values estimated based on person weighing 70 Kg')

    else:
        title3 = f"<h2 style='text-align:center'>⚠️Whoops, Our system did not recognize the food⚠️</h2>"
        st.markdown(title3, unsafe_allow_html=True)
        title4 = f"<h3 style='text-align:center'>Please try another image</h3>"
        st.markdown(title4, unsafe_allow_html=True)

# Header-Section
col1, col2, col3 = st.columns(3)
col2.image('Nutriends_1.png')
st.subheader('Nutrinfo : Find Food Nutrients Through Image')

# Image Upload Section
choose = st.selectbox("Select Input Method", ["Upload an Image", "URL from Web"])

if choose == "Upload an Image":  # If user chooses to upload image
    file = st.file_uploader("Upload an image...", type=["jpg", "png", 'Tiff'])
    if file is not None:
        img = Image.open(file)
else:  # If user chooses to upload image from url
    url = st.text_area("URL", placeholder="Put URL here")
    if url:
        try:  # Try to get the image from the url
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:  # If the url is not valid, show error message
            st.error(
                "Failed to load the image. Please use a different URL or upload an image."
            )

# Predict Button
if img is not None:
    predict = st.button("Find")
    if predict:
        predict_image(img)