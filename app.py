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
    # runs_left = target - score
    # balls_left = 120 - (overs*6)
    # wickets = 10 - wickets
    # crr = score/overs
    # rrr = (runs_left*6)/balls_left

    # input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    # result = pipe.predict_proba(input_df)
    # loss = result[0][0]
    # win = result[0][1]
    # st.sidebar.header(batting_team + "- " + str(round(win*100)) + "%")
    # st.sidebar.header(bowling_team + "- " + str(round(loss*100)) + "%")
    
    temp=['Minor Damage', 'Significant Damage', 'Severe Damage','Highly Fatal']
    
    # result=pipe.predict([[safety_score,days_inspection,safety_complaints,control_metric,turbulence,cabin_temp,acc_type,max_elev,violations,adv]])
    result=pipe.predict([[safety_score,days_inspection,safety_complaints,control_metric,turbulence,50,2,52,1,12]])
    st.sidebar.header(temp[result[0]])

st.header("Airplane Crash Severity Prediction")
# st.header("Number of Accidents Evaluated : ")
# col6,col7,col8=st.columns(3)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Number of accidents monitored :-", "9509", "")
# col2.metric("Rows :-", "9509", "")
# col3.metric("Columns", "12", "")

# CHARTS SHOWING PIE
data2= pd.read_csv('train.csv')

incidents = data2.groupby("Severity")["Accident_ID"].count()
incidents2 = data2.groupby("Turbulence_In_gforces")["Severity"].count()
# plt.bar(incidents.index, incidents.values)
# plt.title("Accidents per category")
# plt.xlabel("Severity")
# plt.ylabel("Number of crashes")
# plt.show()

#second database for charts
df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.strftime('%Y')

df.rename(columns = {'Flight #': 'Flights'}, inplace = True)

by_year = df.groupby('Year')["Flights"].count()

# Temp = df.groupby(df['Year'])[['Year']].count() #Temp is going to be temporary data frame 
# Temp = Temp.rename(columns={"Year": "Count"})

# plt.figure(figsize=(12,6))
# plt.style.use('bmh')
# plt.plot(Temp.index, 'Count', data=Temp, color='blue', marker = ".", linewidth=1)
# plt.xlabel('Year', fontsize=10)
# plt.ylabel('Count', fontsize=10)
# plt.title('Count of accidents by Year', loc='Center', fontsize=14)
# plt.show()

Fatalities = df.groupby('Year')['Fatalities'].count()

st.write("Number of Fatalities by Year")
st.line_chart(Fatalities)

# df.Operator = df.Operator.str.upper()
# df.Operator = df.Operator.replace('A B AEROTRANSPORT', 'AB AEROTRANSPORT')

operator_counts = df['Operator'].value_counts().reset_index().head(10)
operator_counts.columns = ['Operator', 'count']

# create an Altair chart using the operator counts
chart = alt.Chart(operator_counts).mark_bar().encode(
    x=alt.X('count', title='Count'),
    y=alt.Y('Operator', title='Operator')
).properties(
    title='Number of Fatalities by Operator'
)

# display the chart using st.altair_chart()

st.altair_chart(chart, use_container_width=True)




st.write("Number of Airplane Crashes by Year")
st.line_chart(by_year)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write("Number of Airplane Crashes by Year")
        st.line_chart(by_year)
    with col2:
        st.bar_chart(incidents, use_container_width=True)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write("Chart 1")
        st.bar_chart(incidents, use_container_width=True)
    with col2:
        st.write("Chart 2")
        st.bar_chart(incidents, use_container_width=True)



st.bar_chart(incidents, use_container_width=True)

st.line_chart(incidents2)
    

fig = px.pie(incidents, values=incidents.values, names=incidents.index)
st.plotly_chart(fig)


severity_counts = data2.groupby("Severity").count()["Accident_ID"]
fig2 = px.box(data2, x="Severity", y="Accident_ID")
st.plotly_chart(fig2)

    
# def Pie_Graph(target,score,overs,wickets):
#     runs_left = targ - score
#     balls_left = 120 - (overs*6)
#     wickets = 10 - wickets
#     crr = score/overs
#     rrr = (runs_left*6)/balls_left

#     input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

#     result = pipe.predict_proba(input_df)
#     loss = result[0][0]
#     win = result[0][1]
    
        
#     data = pd.DataFrame({
#         'Winning': [batting_team, bowling_team],'Percentage': [round(win*100), round(loss*100)]
#         })

#     # Create pie chart
#     fig = px.pie(data, values='Percentage', names='Winning',title='Pie Chart with Percentage Labels',hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3)

#     # Render chart
#     st.plotly_chart(fig, use_container_width=True)


# # CHARTS SHOWING BAR

# def Bar_Graph(target,score,overs,wickets):
#     runs_left = target - score
#     balls_left = 120 - (overs*6)
#     wickets = 10 - wickets
#     crr = score/overs
#     rrr = (runs_left*6)/balls_left

#     input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

#     result = pipe.predict_proba(input_df)
#     loss = result[0][0]
#     win = result[0][1]
    

#     # Create a sample dataframe
#     data = {
#         'Winning': [batting_team, bowling_team],
#         'Percentage': [round(win*100), round(loss*100)]
#     }
#     df = pd.DataFrame(data)

#     # Set up the bar chart using Altair
#     bars = alt.Chart(df).mark_bar().encode(
#         x='Winning',
#         y='Percentage'
#     )

#     # Set the chart's title and axis labels
#     chart = bars.properties(
#         title='Sample Bar Chart',
#         width=alt.Step(80)
#     )

#     # Display the chart in Streamlit
#     st.altair_chart(chart, use_container_width=True) 



# from streamlit_echarts import st_echarts


# if col6.button("PIE GRAPH"):
#     Pie_Graph(target,score,overs,wickets)

# if col7.button("BAR GRPAH"):
#     Bar_Graph(target,score,overs,wickets)

# option = {
#     "xAxis": {
#         "type": "category",
#         "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
#     },
#     "yAxis": {"type": "value"},
#     "series": [{"data": [820, 932, 901, 934, 1290, 1330, 1320], "type": "line"}],
# }
# st_echarts(
#     options=option, height="400px",
# )