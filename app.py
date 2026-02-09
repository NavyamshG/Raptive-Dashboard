import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="Raptive ROI Decision Engine", layout="wide")

# Custom CSS for metric styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 Strategic Bayesian A/B Engine")
st.caption("Bridging Statistical Rigor with Executive ROI.")

# --- SIDEBAR: BUSINESS & DATA INPUTS ---
with st.sidebar:
    st.header("🕹️ Experiment Controls")
    
    with st.expander("📊 Primary Experiment Data", expanded=True):
        col_a, col_b = st.columns(2)
        clicks_a = col_a.number_input("Control Clicks", value=1000)
        views_a = col_a.number_input("Control Views", value=10000)
        clicks_b = col_b.number_input("Variant Clicks", value=1150)
        views_b = col_b.number_input("Variant Views", value=10500)

    with st.expander("💰 Business Economics", expanded=True):
        avg_order_value = st.number_input("Average Order Value ($)", value=50.0)
        monthly_traffic = st.number_input("Monthly Traffic", value=100000)

    with st.expander("⚙️ Advanced Bayesian Settings", expanded=False):
        st.write("Informative Priors (Avoid starting from zero)")
        baseline_ctr = st.slider("Historical CTR (%)", 0.0, 20.0, 10.0) / 100
        prior_weight = st.number_input("Prior Strength (Effective Samples)", value=100)
        
        st.divider()
        risk_tolerance = st.slider("Risk Tolerance (Max Loss $)", 0, 1000, 100)

# --- CALCULATIONS ---
# Prior Parameters
alpha_prior = prior_weight * baseline_ctr
beta_prior = prior_weight * (1 - baseline_ctr)

# Posterior Samples (Monte Carlo)
N_SAMPLES = 20000
sim_a = np.random.beta(alpha_prior + clicks_a, beta_prior + (views_a - clicks_a), N_SAMPLES)
sim_b = np.random.beta(alpha_prior + clicks_b, beta_prior + (views_b - clicks_b), N_SAMPLES)

# Metrics
prob_b_better = (sim_b > sim_a).mean()
relative_lift = (sim_b - sim_a) / sim_a
expected_uplift = np.median(relative_lift)

# Financial Impact
current_revenue = (clicks_a / views_a) * monthly_traffic * avg_order_value
expected_revenue_delta = expected_uplift * current_revenue
# Expected Loss: If B < A, how much do we lose?
loss = np.maximum(0, (sim_a - sim_b) * monthly_traffic * avg_order_value)
expected_loss = np.mean(loss)

# --- TOP LEVEL KPI DASHBOARD ---
m1, m2, m3, m4 = st.columns(4)

m1.metric("Win Probability", f"{prob_b_better:.1%}", 
          help="Probability the variant is better than control.")
m2.metric("Expected Lift", f"{expected_uplift:+.1%}", 
          help="The most likely percentage change in conversion.")
m3.metric("Est. Monthly Revenue", f"${expected_revenue_delta:+,.0f}", 
          help="Projected dollar impact based on monthly traffic.")

# Decision Logic
is_significant = prob_b_better > 0.95
is_low_risk = expected_loss < risk_tolerance

if is_significant and is_low_risk:
    status = "✅ DEPLOY"
    color = "green"
elif is_significant:
    status = "⚠️ CAUTION"
    color = "orange"
else:
    status = "🛑 GATHER DATA"
    color = "red"

m4.markdown(f"**Decision Status**<br><h2 style='color:{color}; margin-top:0;'>{status}</h2>", unsafe_allow_html=True)

st.divider()

# --- VISUALIZATION TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Density", "📉 Risk Analysis", "🧪 Statistical Rigor"])

with tab1:
    st.subheader("Interactive Conversion Density")
    st.write("Where is the 'Winning Zone'?")
    
    # Plotly Density Map
    fig_dens = go.Figure()
    fig_dens.add_trace(go.Violin(x=sim_a, name='Control', line_color='#95a5a6', side='negative', meanline_visible=True))
    fig_dens.add_trace(go.Violin(x=sim_b, name='Variant', line_color='#2ecc71', side='positive', meanline_visible=True))
    fig_dens.update_layout(xaxis_title="Conversion Rate (CTR)", violinmode='overlay', height=400)
    st.plotly_chart(fig_dens, use_container_width=True)

with tab2:
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.subheader("Financial Risk (Expected Loss)")
        # Gauge Chart for Risk
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = expected_loss,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Expected Loss vs ${risk_tolerance} Cap"},
            gauge = {
                'axis': {'range': [0, risk_tolerance * 2]},
                'bar': {'color': "black"},
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': risk_tolerance},
                'steps': [
                    {'range': [0, risk_tolerance], 'color': "lightgreen"},
                    {'range': [risk_tolerance, risk_tolerance * 2], 'color': "pink"}]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_r:
        st.subheader("Likely Revenue Outcomes")
        revenue_sim = relative_lift * current_revenue
        fig_rev = px.histogram(revenue_sim, nbins=50, color_discrete_sequence=['#2ecc71'], 
                               labels={'value': 'Monthly Revenue Impact ($)'})
        fig_rev.add_vline(x=0, line_dash="dash", line_color="red")
        fig_rev.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_rev, use_container_width=True)

with tab3:
    st.subheader("Data Scientist's Workbench")
    # Technical boxplot for variance check
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=sim_a, name="Control", marker_color='#95a5a6'))
    fig_box.add_trace(go.Box(y=sim_b, name="Variant", marker_color='#2ecc71'))
    fig_box.update_layout(title="CTR Confidence Intervals (Full Distribution)", height=450)
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.info(f"""
    **Statistical Summary:**
    - **Control Median:** {np.median(sim_a):.4%}
    - **Variant Median:** {np.median(sim_b):.4%}
    - **95% HDI (Variant):** {np.percentile(sim_b, 2.5):.4%} to {np.percentile(sim_b, 97.5):.4%}
    """)

st.success("**Leader Summary:** " + (
    f"This variant is expected to generate **${expected_revenue_delta:,.2f}** in additional monthly revenue. "
    f"The statistical risk (Expected Loss) is **${expected_loss:,.2f}**, which is {'within' if is_low_risk else 'outside'} your risk tolerance."
))
