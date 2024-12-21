#Importing Library

#Modeling import
import joblib
# Data Analysis
import numpy as np
import pandas as pd
from scipy.stats import boxcox

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Data Modeling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBRegressor


#Streaming
import streamlit as st


#Page editing
st.set_page_config(
    page_title="Residential Property Prediction Tools!",
    layout="wide"  # Use the 'wide' layout
)
#input frame
req_full = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
       'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'SaleType', 'SaleCondition']
input = pd.DataFrame(columns= req_full)

# Custom HTML and CSS to style the text
st.markdown(
    """
    <style>
    .custom-header {
        font-size: 50px; /* Adjust size as needed */
        font-weight: bold;
    }
    .gold {
        color: gold; /* Gold color for "I" */
    }
    .cardinal {
        color: #8C1515; /* Cardinal color for the rest */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render the styled text

st.image("logo2.png", use_container_width= True)

# Title text
st.write("# Greetings Ames, Iowa citizens! ")

st.write("##### _Welcome to our House Price Prediction tool, designed specifically for the vibrant community of Ames, Iowa. Whether you're a seasoned homeowner looking to assess your property's value or a prospective buyer exploring the market, our accurate and user-friendly tool is here to assist you. Our model leverages advanced machine learning techniques and a comprehensive dataset of historical home sales in Ames to provide reliable price estimates. By inputting key property details, you can gain valuable insights into potential market values and make informed decisions._")
st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 20px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 75px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")

col1, col3, col4 = st.columns([1, 1, 1])

with col1:
    st.header("Did you know? ")
    st.write("Ames, Iowa, is home to a unique architectural style known as the Prairie School. This style, popularized by architects like Frank Lloyd Wright, emphasizes horizontal lines, open floor plans, and a strong connection to the natural environment. Many historic homes in Ames showcase this distinctive architectural heritage, making it a fascinating place for architecture enthusiasts to explore.")
    st.image("prairie_bobshimer.jpg", caption="Prairie style buildings", use_container_width=True)

with open('data_description.txt', 'r') as file:
    content2 = file.read()
with open('datafield.txt', 'r') as file:
    content1 = file.read()

with col3: 
    st.header("Data description")
    st.text_area("", value= content1, height = 500,  )
with col4: 
    st.header("Metadata")
    st.text_area("", value= content2, height= 500)

st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 20px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 75px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")
st.header("Prediction Terminal")
st.text("Note: Before adding any value, Please consult and take a look at attribues description list for each attributes that you would input into our model. ")

c1, c15, c2 = st.columns([1,0.07, 1])
with c1:
       st.subheader("Only input numerical values")
       # Numerical Columns
       input.loc[0, "MSSubClass"] = st.selectbox("MSSubClass: The building class",([None,20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]))
       input.loc[0, "LotFrontage"] = st.slider(
       "LotFrontage: Linear feet of street connected to property", 
       min_value=0, 
       max_value=int(200.0 * 1.5)
       )
       input.loc[0, "LotArea"] = st.slider(
       "LotArea: Lot size in square feet", 
       min_value=0, 
       max_value=int(56600 * 1.5)
       )
       input.loc[0, "OverallQual"] = st.slider(
       "OverallQual: Overall material and finish quality", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "OverallCond"] = st.slider(
       "OverallCond: Overall condition rating", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "YearBuilt"] = st.slider(
       "YearBuilt: Original construction date", 
       min_value=1800, 
       max_value=int(2024)
       )
       input.loc[0, "YearRemodAdd"] = st.slider(
       "YearRemodAdd: Remodel date", 
       min_value=1800, 
       max_value=int(2024)
       )
       input.loc[0, "MasVnrArea"] = st.slider(
       "MasVnrArea: Masonry veneer area in square feet", 
       min_value=0, 
       max_value=int(1290.0 * 1.5)
       )
       input.loc[0, "BsmtFinSF1"] = st.slider(
       "BsmtFinSF1: Type 1 finished square feet", 
       min_value=0, 
       max_value=int(4010.0 * 1.5)
       )
       input.loc[0, "BsmtUnfSF"] = st.slider(
       "BsmtUnfSF: Unfinished square feet of basement area", 
       min_value=0, 
       max_value=int(2140.0 * 1.5)
       )
       input.loc[0, "TotalBsmtSF"] = st.slider(
       "TotalBsmtSF: Total square feet of basement area", 
       min_value=0, 
       max_value=int(5095.0 * 1.5)
       )
       input.loc[0, "1stFlrSF"] = st.slider(
       "1stFlrSF: First Floor square feet", 
       min_value=0, 
       max_value=int(5095 * 1.5)
       )
       input.loc[0, "2ndFlrSF"] = st.slider(
       "2ndFlrSF: Second floor square feet", 
       min_value=0, 
       max_value=int(1862 * 1.5)
       )
       input.loc[0, "GrLivArea"] = st.slider(
       "GrLivArea: Above grade (ground) living area square feet", 
       min_value=0, 
       max_value=int(5095 * 1.5)
       )
       input.loc[0, "BsmtFullBath"] = st.slider(
       "BsmtFullBath: Basement full bathrooms", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "BsmtHalfBath"] = st.slider(
       "BsmtHalfBath: Basement half bathrooms", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "FullBath"] = st.slider(
       "FullBath: Full bathrooms above grade", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "HalfBath"] = st.slider(
       "HalfBath: Half baths above grade", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "BedroomAbvGr"] = st.slider(
       "BedroomAbvGr: Number of bedrooms above basement level", 
       min_value=0, 
       max_value=int(20)
       )
       input.loc[0, "KitchenAbvGr"] = st.slider(
       "KitchenAbvGr: Number of kitchens", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "TotRmsAbvGrd"] = st.slider(
       "TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)", 
       min_value=0, 
       max_value=int(15 * 1.5)
       )
       input.loc[0, "Fireplaces"] = st.slider(
       "Fireplaces: Number of fireplaces", 
       min_value=0, 
       max_value=int(15)
       )
       input.loc[0, "GarageYrBlt"] = st.slider(
       "GarageYrBlt: Year garage was built", 
       min_value=1800, 
       max_value=int(2024)
       )
       input.loc[0, "GarageCars"] = st.slider(
       "GarageCars: Size of garage in car capacity", 
       min_value=0, 
       max_value=int(10)
       )
       input.loc[0, "GarageArea"] = st.slider(
       "GarageArea: Size of garage in square feet", 
       min_value=0, 
       max_value=int(1488.0 * 1.5)
       )
       input.loc[0, "WoodDeckSF"] = st.slider(
       "WoodDeckSF: Wood deck area in square feet", 
       min_value=0, 
       max_value=int(1424 * 1.5)
       )
       input.loc[0, "OpenPorchSF"] = st.slider(
       "OpenPorchSF: Open porch area in square feet", 
       min_value=0, 
       max_value=int(742 * 1.5)
       )
with c15:
     st.text("  ")

with c2:   
       st.subheader("Only input text values according to the metadata below")
       input.loc[0, "MSZoning"] = st.selectbox("MSZoning: The general zoning classification", (None, "A", "C", "FV", "I", "RH", "RL", "RP", "RM"))
       input.loc[0, "Street"] = st.selectbox("Street: Type of road access", (None, "Grvl", "Pave"))
       input.loc[0, "LotShape"] = st.selectbox("LotShape: General shape of property", (None, "Reg", "IR1", "IR2", "IR3"))
       input.loc[0, "LandContour"] = st.selectbox("LandContour: Flatness of the property", (None, "Lvl", "Bnk", "HLS", "Low"))
       input.loc[0, "Utilities"] = st.selectbox("Utilities: Type of utilities available", (None, "AllPub", "NoSewr", "NoSeWa", "ELO"))
       input.loc[0, "LotConfig"] = st.selectbox("LotConfig: Lot configuration", (None, "Inside", "Corner", "CulDSac", "FR2", "FR3"))
       input.loc[0, "LandSlope"] = st.selectbox("LandSlope: Slope of property", (None, "Gtl", "Mod", "Sev"))
       input.loc[0, "Neighborhood"] = st.selectbox("Neighborhood: Physical locations within Ames city limits", (None, "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"))
       input.loc[0, "Condition1"] = st.selectbox("Condition1: Proximity to main road or railroad", (None, "Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"))
       input.loc[0, "Condition2"] = st.selectbox("Condition2: Proximity to main road or railroad (if a second is present)", (None, "Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"))
       input.loc[0, "BldgType"] = st.selectbox("BldgType: Type of dwelling", (None, "1Fam", "2FmCon", "Duplex", "TwnhsE", "Twnhs"))
       input.loc[0, "HouseStyle"] = st.selectbox("HouseStyle: Style of dwelling", (None, "1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"))
       input.loc[0, "RoofStyle"] = st.selectbox("RoofStyle: Type of roof", (None, "Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"))
       input.loc[0, "RoofMatl"] = st.selectbox("RoofMatl: Roof material", (None, "ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"))
       input.loc[0, "Exterior1st"] = st.selectbox("Exterior1st: Exterior covering on house", (None, "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"))
       input.loc[0, "Exterior2nd"] = st.selectbox("Exterior2nd: Exterior covering on house (if more than one material)", (None, "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"))
       input.loc[0, "MasVnrType"] = st.selectbox("MasVnrType: Masonry veneer type", (None, "BrkCmn", "BrkFace", "CBlock", "None", "Stone"))
       input.loc[0, "ExterQual"] = st.selectbox("ExterQual: Exterior material quality", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "ExterCond"] = st.selectbox("ExterCond: Present condition of the material on the exterior", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "Foundation"] = st.selectbox("Foundation: Type of foundation", (None, "BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"))
       input.loc[0, "BsmtQual"] = st.selectbox("BsmtQual: Height of the basement", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "BsmtCond"] = st.selectbox("BsmtCond: General condition of the basement", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "BsmtExposure"] = st.selectbox("BsmtExposure: Walkout or garden level basement walls", (None, "Gd", "Av", "Mn", "No", "NA"))
       input.loc[0, "BsmtFinType1"] = st.selectbox("BsmtFinType1: Quality of basement finished area", (None, "GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"))
       input.loc[0, "BsmtFinType2"] = st.selectbox("BsmtFinType2: Quality of second finished area (if present)", (None, "GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"))
       input.loc[0, "Heating"] = st.selectbox("Heating: Type of heating", (None, "Floor", "GasA", "GasW", "Grav", "OthW", "Wall"))
       input.loc[0, "HeatingQC"] = st.selectbox("HeatingQC: Heating quality and condition", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "CentralAir"] = st.selectbox("CentralAir: Central air conditioning", (None, "Y", "N"))
       input.loc[0, "Electrical"] = st.selectbox("Electrical: Electrical system", (None, "SBrkr", "FuseA", "FuseF", "FuseP", "Mix"))
       input.loc[0, "KitchenQual"] = st.selectbox("KitchenQual: Kitchen quality", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "Functional"] = st.selectbox("Functional: Home functionality rating", (None, "Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"))
       input.loc[0, "FireplaceQu"] = st.selectbox("FireplaceQu: Fireplace quality", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "GarageType"] = st.selectbox("GarageType: Garage location", (None, "2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"))
       input.loc[0, "GarageFinish"] = st.selectbox("GarageFinish: Interior finish of the garage", (None, "Fin", "RFn", "Unf", "NA"))
       input.loc[0, "GarageQual"] = st.selectbox("GarageQual: Garage quality", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "GarageCond"] = st.selectbox("GarageCond: Garage condition", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "PavedDrive"] = st.selectbox("PavedDrive: Paved driveway", (None, "Y", "P", "N"))
       input.loc[0, "SaleType"] = st.selectbox("SaleType: Type of sale", (None, "WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"))
       input.loc[0, "SaleCondition"] = st.selectbox("SaleCondition: Condition of sale", (None, "Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"))


st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 5px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 5px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")
#Prediction Process------------------------------------------------------------------------------------------------------------------------------------------------

#loading pipeline
pipeline = joblib.load('pipelinexgb.pkl')
#prediction
pred = pipeline.predict(input)

#Prediction Result Terminal------------------------------------------------------------------------------------------------------------------------------------
st.header("Prediction Result")
st.subheader("Summary of your property attributes")
st.dataframe(input)
st.subheader("estimate is " + str(pred))


st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 20px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 75px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")

#Result Comparison------------------------------------------------------
st.header("Result Comparison")
comparison = pd.read_csv("train.csv")
newtrain = comparison[input.columns]
newtrain["SalePrice"] = comparison["SalePrice"]
newtrain["From"] = "Others"
input["From"] = "Your Result"
input["SalePrice"] = pred
pdata = pd.concat([newtrain,input], axis= 0)


req_num = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
pdata["size"] = pdata["From"].apply(lambda x: 5 if x == "Your Result" else 2)

# Display in Streamlit
selected_columns = st.multiselect(
    "Select columns to plot",
    options=req_num,
    default=req_num[:4]  # Default to first 3 columns if nothing is selected
)

# Create columns for layout in Streamlit
laycol = st.columns(2)

# Loop through selected columns and create a scatter plot
for i, col in enumerate(selected_columns):
    # Ensure the column exists in the DataFrame before plotting
    if col in pdata.columns:
        # Add a size column for demonstration (you can modify this logic as needed)
        pdata["size"] = pdata["From"].apply(lambda x: 5 if x == "Your Result" else 2)

        # Create scatter plot for the current column
        fig = px.scatter(
            pdata,
            x=col,
            y="SalePrice",
            color="From",  # Color based on 'From' subcategory
            title=f"Scatter Plot of SalePrice by {col}",
            labels={col: col, "From": "Category"},
            color_discrete_sequence=["#C41E3A", "#FFD700"],  # Custom color palette
            size="size",  # Size based on 'From' subcategory
            size_max=15  # Maximum size of dots
        )

        # Remove the white outline by setting line width to 0
        fig.update_traces(marker=dict(line=dict(width=0)))

        # Display the plot in the appropriate column
        laycol[i % 2].plotly_chart(fig)  # This ensures the plots are distributed across 2 columns