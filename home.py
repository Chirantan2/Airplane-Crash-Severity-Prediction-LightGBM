import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import pandas as pd

# icon and title
st.set_page_config(page_title="Airplane Crash Severity Prediction", page_icon=":line_chart:",initial_sidebar_state="expanded")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



# Add some CSS styles to the title
st.markdown(
    f"""
    <style>
        h1 {{
            color: #0072B2;
            text-align: center;
        }}
    </style>
    """,
    unsafe_allow_html=True
)



# Add some CSS styles to the selectbox
st.markdown(
    f"""
    <style>
        .stSelectbox {{
            border-radius:10px;
            border: none;
            padding: 0.5rem;
            font-size: 1rem;
        }}

        .stSelectbox:hover {{
            background-color:Black;
        }}

        .stSelectbox:focus {{
            outline: none;
            box-shadow: none;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# pipe = pickle.load(open("pipe.pkl", 'rb'))


# TITLE OF PAGE
st.sidebar.markdown('<h1>Airplane Crash Severity Prediction</h1>', unsafe_allow_html=True)

from streamlit_echarts import st_echarts


option = {
    "xAxis": {
        "type": "category",
        "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    },
    "yAxis": {"type": "value"},
    "series": [{"data": [820, 932, 901, 934, 1290, 1330, 1320], "type": "line"}],
}
st_echarts(
    options=option, height="400px",
)