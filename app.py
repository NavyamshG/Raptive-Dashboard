import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

st.set_page_config(page_title="Raptive Advanced Decision Engine", layout="wide")

st.title("🧪 Advanced Bayesian A/B Decision Engine")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Experiment Data")
with st.sidebar.expander("Control Settings", expanded=True):
    clicks_a = st.number_input("Control Conversions", value=100)
    views_a = st.number_input("Control Impressions", value=1000)
with st.sidebar.expander("Variant Settings", expanded=True):
    clicks_b = st.number_input("Variant Conversions", value=125)
    views_b = st.number_input("Variant Impressions", value=1050)

# --- STATISTICAL CALCULATIONS ---
# Generate distributions for plotting
x = np.linspace(0, 0.25, 500)
y_a = beta.pdf(x, clicks_a + 1, views_a - clicks_a + 1)
y_b = beta.pdf(x, clicks_b + 1, views_b - clicks_b + 1)

# Monte Carlo Simulation for Probability & Relative Lift
sim_a = np.random.beta(clicks_a + 1, views_a - clicks_a + 1, 10000)
sim_b = np.random.beta(clicks_b + 1, views_b - clicks_b + 1, 10000)
prob_b_better = (sim_b > sim_a).mean()
relative_lift = (sim_b - sim_a) / sim_a

# --- MAIN DASHBOARD ---
m1, m2, m3 = st.columns(3)
m1.metric("Prob. Variant is Better", f"{prob_b_better:.1%}")
m2.metric("Mean Relative Lift", f"{(sim_b.mean()/sim_a.mean())-1:+.2%}")
m3.metric("Decision Status", "Deploy" if prob_b_better > 0.95 else "Collect Data")

st.divider()

# --- CHART SECTION ---
tab1, tab2 = st.tabs(["📊 Confidence Distributions", "📈 Risk & Lift Analysis"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Performance Density")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y_a, label="Control", color='#95a5a6', lw=3)
        ax.fill_between(x, 0, y_a, alpha=0.2, color='#95a5a6')
        ax.plot(x, y_b, label="Variant", color='#2ecc71', lw=3)
        ax.fill_between(x, 0, y_b, alpha=0.2, color='#2ecc71')
        ax.set_xlabel("Conversion Rate (CTR)")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Range of Uncertainty")
        # Boxplot to show the spread of the CTR
        fig_box, ax_box = plt.subplots(figsize=(5, 7.3))
        sns.boxplot(data=[sim_a, sim_b], palette=['#95a5a6', '#2ecc71'], ax=ax_box)
        ax_box.set_xticklabels(['Control', 'Variant'])
        ax_box.set_title("CTR Confidence Intervals")
        st.pyplot(fig_box)



with tab2:
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.subheader("📈 Decision Confidence")
        # Probability Gauge for immediate visual "Go/No-Go"
        import plotly.graph_objects as go
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_b_better * 100,
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 70], 'color': "#ff4b4b"},   # Red: High Risk
                    {'range': [70, 95], 'color': "#ffa500"},  # Orange: Warning
                    {'range': [95, 100], 'color': "#00cc96"}  # Green: Safe
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'value': 95}
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.subheader("🚀 Business Impact Summary")
        
        # Big Bold Metrics
        st.metric("Most Likely Lift", f"{np.median(relative_lift):+.1%}")
        
        # Logic-driven status boxes
        worst_case = np.percentile(relative_lift, 5)
        
        if prob_b_better >= 0.95:
            st.success(f"✅ **Safe to Deploy**: Confidence is high ({prob_b_better:.1%}). Even the worst-case scenario is likely manageable.")
        elif prob_b_better >= 0.80:
            st.warning(f"⚠️ **Directional Win**: High probability of success, but hasn't hit 95% certainty. Worst-case: {worst_case:.1%} drop.")
        else:
            st.error(f"🛑 **Inconclusive**: Keep the test running. There is too much overlap in the 'Range of Uncertainty' (Boxplot).")

        st.info(f"**Insight:** We are {prob_b_better:.1%} sure that the Variant will outperform the Control.")
