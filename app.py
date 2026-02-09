import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Raptive Revenue Forecaster", layout="wide")

st.title("🎲 Raptive Monte Carlo: Ad Revenue Forecaster")
st.markdown("""
### From Chaos to Certainty
In advertising, daily revenue is volatile. This tool uses **Monte Carlo Simulation** to model 1,000s of possible monthly outcomes for a creator.
It demonstrates the **Central Limit Theorem**: how the sum of many independent daily variables (Traffic & CPM) results in a stable Normal Distribution of total monthly revenue.
""")

# Sidebar for Business Logic
st.sidebar.header("📈 Forecast Inputs")
avg_traffic = st.sidebar.number_input("Avg. Daily Pageviews", value=50000, step=1000)
traffic_volatility = st.sidebar.slider("Daily Traffic Volatility (%)", 0, 50, 15)

avg_cpm = st.sidebar.slider("Target CPM ($)", 5.0, 50.0, 22.0)
cpm_volatility = st.sidebar.slider("CPM Daily Volatility ($)", 0.5, 10.0, 3.0)

num_simulations = st.sidebar.selectbox("Precision (Simulations)", [500, 1000, 2000], index=1)

# The Math: Monte Carlo Logic
@st.cache_data
def run_simulation(traffic, t_vol, cpm, cpm_vol, sims):
    results = []
    days = 30
    for _ in range(sims):
        # Simulate 30 individual days of noise
        daily_traffic = np.random.normal(traffic, traffic * (t_vol/100), days)
        daily_cpm = np.random.normal(cpm, cpm_vol, days)
        # Calculate 30-day sum
        monthly_rev = np.sum((daily_traffic / 1000) * daily_cpm)
        results.append(monthly_rev)
    return np.array(results)

rev_data = run_simulation(avg_traffic, traffic_volatility, avg_cpm, cpm_volatility, num_simulations)

# Visualizing the "Wow"
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("The Revenue Probability Cloud")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(rev_data, kde=True, color="#1f77b4", ax=ax, bins=30)
    plt.axvline(np.mean(rev_data), color='red', linestyle='--', label=f'Expected: ${np.mean(rev_data):,.0f}')
    plt.title("Distribution of 1,000+ Potential Monthly Outcomes")
    plt.xlabel("Total Monthly Revenue ($)")
    plt.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Risk Analysis")
    p5 = np.percentile(rev_data, 5)
    p50 = np.percentile(rev_data, 50)
    p95 = np.percentile(rev_data, 95)
    
    st.metric("Conservative (5% chance)", f"${p5:,.0f}")
    st.metric("Expected (Median)", f"${p50:,.0f}")
    st.metric("Optimistic (95% chance)", f"${p95:,.0f}")
    
    st.write(f"""
    **Business Insight:** There is a **90% statistical probability** that this creator will earn between 
    **${p5:,.0f}** and **${p95:,.0f}** next month.
    """)

st.info("💡 **Why this matters for Raptive:** We help creators turn unpredictable traffic into predictable income. This simulation shows that while any single day is a gamble, the aggregate month follows a stable statistical law, allowing for confident financial planning.")
