import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

st.set_page_config(
    page_title="Energy Consumption", page_icon="📊", initial_sidebar_state="expanded"
)

def boxplot(x):
    fig, ax = plt.subplots()
    ax.boxplot(x, vert = False)
    st.write(fig)

st.write("# Prédiction de la consommation d'energie")

with st.form(key="my_form"):
    strategy = st.selectbox(
        "Stratégie",
        options=["Stratégie 1","Stratégie 2"],
        help="Selectionner une stratégie"
    )
    modele = st.selectbox(
        "Modèle",
        options=["Linear Simple","Linear Ridge"],
        help="Selectionner un modèle"
    )
    submit_button = st.form_submit_button(label="Submit")

bench = pd.read_excel("benchmark_cleaned_1.xlsx")
bench=bench.drop(bench[bench["SiteEnergyUse(kBtu)"]>0.7*10**8].index)
bench = bench.rename(columns = {"PropertyAge":"a","NumberofBuildings":"nb","NumberofFloors":"nf"})
bench["a^2"] = bench["a"]**2
bench["a^3"] = bench["a"]**3
bench["nb^2"] = bench["nb"]**2
bench["nb^3"] = bench["nb"]**3
bench["nf^2"] = bench["nf"]**2
bench["nf^3"] = bench["nf"]**3

st.dataframe(bench)
#"a^3","nb^3","nf^3"
X = bench[["a","a^2","nb","nb^2","nf","nf^2"]]
y = bench["SiteEnergyUse(kBtu)"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=45)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

st.metric(label = "RMSE",value = mean_squared_error(y_pred,y_test, squared=False))

l = st.select_slider(label = "lambda", options = np.logspace(1,6,6))

ridge = Ridge(alpha=l)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
st.metric(label = "RMSE", value = mean_squared_error(y_pred, y_test, squared=False))


boxplot(bench["SiteEnergyUse(kBtu)"])