import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import pandas as pd

# icon and title
st.set_page_config(page_title="IPL Win Predictor", page_icon=":bar_chart:",layout="wide",initial_sidebar_state="expanded")

# ---- HIDE STREAMLIT STYLE ----
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)



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





teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open("pipe.pkl", 'rb'))


# TITLE OF PAGE
st.sidebar.markdown('<h1>Match Win Predictor</h1>', unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    batting_team = st.sidebar.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.sidebar.selectbox('Select the bowling team', (teams))

selected_city = st.sidebar.selectbox('Select host city', sorted(cities))

target = st.sidebar.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.sidebar.number_input('Score')
with col4:
    overs = st.sidebar.number_input('Overs completed')
with col5:
    wickets = st.sidebar.number_input('Wickets out')


col6,col7 = st.sidebar.columns(2)


#  PROBABILITY SHOWING
if col6.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")
    


# CHARTS SHOWING

def my_chart(target,score,overs,wickets):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    
    # cloumn
    # col8,col9 = st.sidebar.columns(2)

    ##
    # Pie Plot
    ##

    # if col8.button("PIE CHART"):
        
    data = pd.DataFrame({
        'Winning': [batting_team, bowling_team],'Percentage': [round(win*100), round(loss*100)]
        })

    # Create pie chart
    fig = px.pie(data, values='Percentage', names='Winning',title='Pie Chart with Percentage Labels',hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3)

    # Render chart
    st.plotly_chart(fig, use_container_width=True)


    ##
    # Bar Plot
    ##

    # Create a sample dataframe
    data = {
        'Winning': [batting_team, bowling_team],
        'Percentage': [round(win*100), round(loss*100)]
    }
    df = pd.DataFrame(data)

    # Set up the bar chart using Altair
    bars = alt.Chart(df).mark_bar().encode(
        x='Winning',
        y='Percentage'
    )

    # Set the chart's title and axis labels
    chart = bars.properties(
        title='Sample Bar Chart',
        width=alt.Step(80)
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)  

if col7.button("Click Here For Charts"):
    my_chart(target,score,overs,wickets)

