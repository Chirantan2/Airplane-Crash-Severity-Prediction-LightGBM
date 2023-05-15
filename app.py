import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import pandas as pd

# icon and title
st.set_page_config(page_title="Crash Severity Prediction", page_icon=":bar_chart:",initial_sidebar_state="expanded")

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

pipe = pickle.load(open("trained_model.sav", 'rb'))


# TITLE OF PAGE
st.sidebar.markdown('<h1>Airplane Crash Severity Predictor</h1>', unsafe_allow_html=True)

# col1, col2 = st.columns(2)


# target = st.sidebar.number_input('Target')

col1,col2,col3,col4,col5,col6,col7,col8 ,col9, col10 = st.columns(10)

with col1:
    safety_score = st.sidebar.number_input('Safety Score')
with col2:
    days_inspection = st.sidebar.number_input('Days Since Inspection')
with col3:
    safety_complaints = st.sidebar.number_input('Total Safety Complaints')
with col4:
    control_metric = st.sidebar.number_input('Control Metric')
with col5:
    turbulence = st.sidebar.number_input('Turbulence')
# with col6:
#     cabin_temp = st.sidebar.number_input('Total Safety Complaints')
# with col7:
#     acc_type = st.sidebar.number_input('Accient Type')
# with col8:
#     max_elev = st.sidebar.number_input('Max Eelevation')
# with col9:
#     violations = st.sidebar.number_input('Violations')
# with col10:
#     adv = st.sidebar.number_input('Adverse Weather Metric')


#  PROBABILITY SHOWING
if st.sidebar.button('Predict Probability'):
    
    temp=['Minor Damage', 'Significant Damage', 'Severe Damage','Highly Fatal']
    
    # result=pipe.predict([[safety_score,days_inspection,safety_complaints,control_metric,turbulence,cabin_temp,acc_type,max_elev,violations,adv]])
    result=pipe.predict([[safety_score,days_inspection,safety_complaints,control_metric,turbulence,50,2,52,1,12]])
    st.sidebar.header(temp[result[0]-1])

st.header("Airplane Crash Dashboard")
# st.header("Number of Accidents Evaluated : ")
# col6,col7,col8=st.columns(3)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown('<br>',unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Number of accidents monitored :-", "9509", "")
# col2.metric("Rows :-", "9509", "")
# col3.metric("Columns", "12", "")
st.markdown('<br>',unsafe_allow_html=True)

# CHARTS SHOWING PIE
data2= pd.read_csv('train.csv')

incidents = data2.groupby("Severity")["Accident_ID"].count()
incidents2 = data2.groupby("Turbulence_In_gforces")["Severity"].count()

#second database for charts
df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.strftime('%Y')

df.rename(columns = {'Flight #': 'Flights'}, inplace = True)

by_year = df.groupby('Year')["Flights"].count()



Fatalities = df.groupby('Year')['Fatalities'].count()

st.write("Number of Fatalities by Year")
st.line_chart(Fatalities)

# df.Operator = df.Operator.str.upper()
# df.Operator = df.Operator.replace('A B AEROTRANSPORT', 'AB AEROTRANSPORT')

operator_counts = df['Operator'].value_counts().reset_index().head(10)
operator_counts.columns = ['Operator', 'count']

st.write("Number of Fatalities by Operator")

# create an Altair chart using the operator counts
chart = alt.Chart(operator_counts).mark_bar().encode(
    x=alt.X('count', title='Count'),
    y=alt.Y('Operator', title='Operator')
).properties(
    # title='Number of Fatalities by Operator'
)

# display the chart using st.altair_chart()

st.altair_chart(chart, use_container_width=True)


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write("Number of Airplane Crashes by Year")
        st.line_chart(by_year)
    with col2:
        st.write("Number of Airplane Crashes by Severity")
        st.bar_chart(incidents, use_container_width=True)

    
st.write("Overview of Severity")
fig = px.pie(incidents, values=incidents.values, names=incidents.index)
st.plotly_chart(fig)


st.write("Effect of Turbulence on Severity")
st.line_chart(incidents2)

