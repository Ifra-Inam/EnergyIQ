import streamlit as st 
import pandas as pd
import joblib 
import os 
import plotly.graph_objects as go 
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY")) # creates groq object for client use

@st.cache_resource 
def load_models():
    models = joblib.load("energy_load_models.pkl")
    return models['heating'], models['cooling']

@st.cache_data
def load_dataset():
    """Load dataset to use for calculating reference standards of energy loads"""
    return pd.read_csv("energy_data.csv")

def get_dataset_stats():
    """Calculate dataset statistics"""
    
    df = load_dataset()
    
    return {
        'avg_heating': df.iloc[:,-2].mean(),
        'avg_cooling': df.iloc[:,-1].mean(),
        
        'min_heating': df.iloc[:,-2].min(),
        'min_cooling': df.iloc[:,-1].min(),
        
        'max_heating': df.iloc[:,-2].max(),
        'max_cooling': df.iloc[:,-1].max()
    }

def comparison_chart(heating_pred, cooling_pred):
    """Compare predicted loads against dataset standards"""

    stats = get_dataset_stats()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name = "Heating Load",
        x = ["Your Building", "Average Case", "Best Case", "Worst Case"],
        y = [heating_pred, stats['avg_heating'], stats['min_heating'], stats['max_heating']],
        marker_color = '#3b96eb', 
    ))

    fig.add_trace(go.Bar(
        name = "Cooling Load",
        x = ["Your Building", "Average Case", "Best Case", "Worst Case"],
        y = [cooling_pred, stats['avg_cooling'], stats['min_cooling'], stats['max_cooling']],
        marker_color = '#3abf37', 
    ))

    fig.update_layout(
        title = "Energy Loads Comparison",
        xaxis_title = "Building Type",
        yaxis_title = "Energy Load (kWh/m²)",
        barmode = "group", 
        height = 400,
        plot_bgcolor = 'white',
        paper_bgcolor = 'white',
        xaxis = dict(
            showline = True,
            linewidth = 1,
            linecolor = "#868686",
            mirror = True
        ),
        yaxis = dict(
            showline = True,
            linewidth = 1,
            linecolor = '#868686',
            mirror = True
        )
    )

    return fig

def get_llm_insights(heating_pred, cooling_pred, input_features):

    stats = get_dataset_stats()

    prompt = f"""You are an energy efficiency consultant. Based on these building predictions: 

    Heating Load: {heating_pred:.2f} kWh/m²
    Cooling Load: {cooling_pred:.2f} kWh/m²

    Dataset Statistics:
    - Average Heating Load: {stats['avg_heating']:.2f} kWh/m²
    - Average Cooling Load: {stats['avg_cooling']:.2f} kWh/m²
    - Most Efficient Heating Load (lowest): {stats['min_heating']:.2f} kWh/m²
    - Most Efficient Cooling Load (lowest): {stats['min_cooling']:.2f} kWh/m²
    - Least Efficient Heating Load (highest): {stats['max_heating']:.2f} kWh/m²
    - Least Efficient Cooling Load (highest): {stats['max_cooling']:.2f} kWh/m²

    Features of Building: 
    - Relative Compactness: {input_features[0]}
    - Surface Area: {input_features[1]} m²
    - Wall Area: {input_features[2]} m²
    - Roof Area: {input_features[3]} m²
    - Overall Height: {input_features[4]} m
    - Orientation: {input_features[5]}°
    - Glazing Area: {input_features[6]} m²
    - Glazing Area Distribution: {input_features[7]}

    Provide:
    1. Brief interpretation (are these loads high/low?)
    2. Top 3 actionable recommendations to improve energy efficiency
    3. Estimated potential savings

    Keep it concise and practical."""   

    try: 
        chat_completion = client.chat.completions.create(
            messages = [
                {"role": "user", "content": prompt}
            ],
            model = "llama-3.1-8b-instant", 
            temperature = 0.7, # determines how random the output should be 
        )
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"LLM service unavailable: {str(e)}"
    
def main():

    st.set_page_config(
        page_title="EnergyIQ",
        page_icon="⚡",
    )
    
    st.title("Welcome to EnergyIQ 💡")
    st.write("Predict the heating and cooling loads of your building and receive AI insights.")

    heating_model, cooling_model = load_models()

    st.header("Buidling Details")

    col1, col2 = st.columns(2)

    with col1: 
        x1 = st.slider("Relative Compactness", 0.62, 0.98, 0.62)
        x2 = st.number_input("Surface Area (m²)", 514.0, 808.0, 514.5)
        x3 = st.number_input("Wall Area (m²)", 245.0, 416.0, 245.0)
        x4 = st.number_input("Roof Area (m²)", 110.0, 220.0, 110.0)

    with col2:
        x5 = st.number_input("Overall Height (m)", 3.5, 7.0, 3.5)
        x6 = st.selectbox("Orientation", [2, 3, 4, 5])
        x7 = st.selectbox("Glazing Area (m²)", [0.0, 0.10, 0.25, 0.40])
        x8 = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

    if st.button("Predict Energy Loads", type="primary"):
        input_data = pd.DataFrame([[x1, x2, x3, x4, x5, x6, x7, x8]])
        heating_pred = heating_model.predict(input_data)[0]
        cooling_pred = cooling_model.predict(input_data)[0]

        # Displays predictions
        st.header("Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Heating Load 🔥", f"{heating_pred:.2f} kWh/m²")
        col2.metric("Cooling Load ❄️", f"{cooling_pred:.2f} kWh/m²")
        col3.metric("Total Load ⚡", f"{heating_pred + cooling_pred:.2f} kWh/m²")

        # Displays chart
        st.subheader("Comparison Against Standards")
        comparison = comparison_chart(heating_pred, cooling_pred)
        st.plotly_chart(comparison, use_container_width=True)

        # Displays AI recs
        st.header("AI Recommendations")
        with st.spinner("Generating insights..."):
            insights = get_llm_insights(
                heating_pred, 
                cooling_pred, 
                [x1, x2, x3, x4, x5, x6, x7, x8]
            )
            st.write(insights)

if __name__ == "__main__":
    main()