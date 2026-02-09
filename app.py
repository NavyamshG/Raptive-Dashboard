import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

st.set_page_config(page_title="Raptive A/B Decision Engine", layout="wide")

st.title("🧪 Bayesian A/B Test Simulator")
st.markdown("""
### Optimization Intelligence
In ad-tech, we don't just want to know if a change worked; we want to know the **probability** that it's better. 
This dashboard uses **Bayesian Inference** to compare two ad layouts (Control vs. Variant).
""")

# Sidebar: Simulation Controls
st.sidebar.header("🕹️ Live Experiment Controls")
st.sidebar.subheader("Baseline (Control)")
clicks_a = st.sidebar.number_input("Control: Clicks/Conversions", value=100, step=10)
views_a = st.sidebar.number_input("Control: Total Impressions", value=1000, step=100)

st.sidebar.subheader("New Layout (Variant)")
clicks_b = st.sidebar.number_input("Variant: Clicks/Conversions", value=125, step=10)
views_b = st.sidebar.number_input("Variant: Total Impressions", value=1050, step=100)

# Statistical Calculation
# Beta distribution is used as a conjugate prior for Bernoulli trials (clicks)
# Parameters: alpha = successes + 1, beta = failures + 1
x = np.linspace(0, 0.25, 500)
y_a = beta.pdf(x, clicks_a + 1, views_a - clicks_a + 1)
y_b = beta.pdf(x, clicks_b + 1, views_b - clicks_b + 1)

# Probability B > A via Monte Carlo simulation
sim_a = np.random.beta(clicks_a + 1, views_a - clicks_a + 1, 10000)
sim_b = np.random.beta(clicks_b + 1, views_b - clicks_b + 1, 10000)
prob_b_better = (sim_b > sim_a).mean()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Statistical Confidence Curves")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_a, label=f"Control (CTR: {clicks_a/views_a:.2%})", lw=3, color='#95a5a6')
    ax.fill_between(x, 0, y_a, alpha=0.2, color='#95a5a6')
    ax.plot(x, y_b, label=f"Variant (CTR: {clicks_b/views_b:.2%})", lw=3, color='#2ecc71')
    ax.fill_between(x, 0, y_b, alpha=0.2, color='#2ecc71')
    
    plt.title("Comparison of Performance Distributions")
    plt.xlabel("Click-Through Rate (CTR)")
    plt.ylabel("Density (Confidence)")
    plt.legend()
    st.pyplot(fig)

with col2:
    st.subheader("The Verdict")
    st.metric("Prob. Variant beats Control", f"{prob_b_better:.1%}")
    
    lift = ((clicks_b/views_b) / (clicks_a/views_a)) - 1
    st.metric("Estimated Revenue Lift", f"{lift:+.1%}")

    if prob_b_better > 0.95:
        st.success("✅ Statistically Significant: Deploy Variant!")
    elif prob_b_better < 0.05:
        st.error("❌ Statistically Significant: Stick with Control.")
    else:
        st.warning("⏳ Inconclusive: Keep collecting data.")

st.info("""
**Why this is "Wow":** Traditional p-values are hard to explain to creators. 
Bayesian probability (e.g., 'There is a 98% chance this ad makes more money') 
is intuitive, actionable, and reflects how top-tier ad-tech companies make decisions.
""")
