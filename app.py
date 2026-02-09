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
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribution of Relative Lift")
        # This shows how MUCH better B is likely to be
        fig_lift, ax_lift = plt.subplots()
        sns.histplot(relative_lift, kde=True, color="#2ecc71", ax=ax_lift)
        ax_lift.axvline(0, color='red', linestyle='--')
        ax_lift.set_title("How much better is the Variant?")
        ax_lift.set_xlabel("Percent Lift over Control")
        st.pyplot(fig_lift)
    
    with c2:
        st.subheader("🚀 Leadership Decision Dashboard")
        
        # Calculate Key Leadership Metrics
        prob_win = prob_b_better
        avg_uplift = np.median(relative_lift)
        worst_case = np.percentile(relative_lift, 5)
        
        # 1. Big Metric Cards for Scannability
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Confidence in Variant", f"{prob_win:.1%}", 
                    help="Probability that the Variant is genuinely better.")
        kpi2.metric("Expected Lift", f"{avg_uplift:+.1%}", 
                    help="The most likely performance increase.")

        st.divider()

        # 2. Status-Based Alerts
        if worst_case < 0:
            st.warning(f"⚠️ **Caution:** Potential risk detected. Worst-case scenario is a **{abs(worst_case):.1%}** drop. Consider more data.")
        else:
            st.success(f"✅ **High Confidence:** Safe to deploy. Even the worst-case is a **{worst_case:.1%}** gain.")

        # 3. The "Elevator Pitch" (Fixed Syntax)
        st.info(f"""
        **The Bottom Line:**
        We are **{prob_win:.1%}** confident that this change will improve results. 
        If deployed, we expect a **{avg_uplift:.1%}** improvement in conversion efficiency.
        """)
