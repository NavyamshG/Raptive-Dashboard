import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

# --- CONFIG ---
st.set_page_config(page_title="Raptive Decision Engine", layout="wide")

st.title("🏆 Executive A/B Decision Engine")
st.caption(f"Strategy & ROI Dashboard | Bayesian Evolution Mode")

# --- SIDEBAR: BUSINESS DATA ---
with st.sidebar:
    st.header("📊 Experiment Data")
    with st.expander("Current Test Data", expanded=True):
        col_a, col_b = st.columns(2)
        clicks_a = col_a.number_input("Control Clicks", value=100)
        views_a = col_a.number_input("Control Views", value=1000)
        clicks_b = col_b.number_input("Variant Clicks", value=130)
        views_b = col_b.number_input("Variant Views", value=1100)
    
    with st.expander("⚙️ Historical Prior (Baseline)", expanded=True):
        st.write("Incorporate existing business knowledge.")
        hist_ctr = st.slider("Historical CTR (%)", 0.0, 30.0, 10.0) / 100
        prior_weight = st.number_input("Prior Strength (Effective Views)", value=500, 
                                      help="Higher weight makes it harder for new data to move the needle.")

    with st.expander("💰 Financials", expanded=True):
        val_per_conv = st.number_input("Value per Click ($)", value=50.0)
        monthly_traffic = st.number_input("Monthly Visitors", value=100000)

# --- BAYESIAN ENGINE ---
# Calculate Prior Parameters (Alpha/Beta)
alpha_p = prior_weight * hist_ctr
beta_p = prior_weight * (1 - hist_ctr)

# Monte Carlo Simulation using the Prior
# Instead of +1, we use alpha_p and beta_p to "anchor" the results
sim_a = np.random.beta(clicks_a + alpha_p, (views_a - clicks_a) + beta_p, 10000)
sim_b = np.random.beta(clicks_b + alpha_p, (views_b - clicks_b) + beta_p, 10000)

prob_b_better = (sim_b > sim_a).mean()
relative_lift = (sim_b - sim_a) / sim_a
avg_lift = np.median(relative_lift)

# Financial Metrics
annual_traffic = monthly_traffic * 12
current_annual_rev = (clicks_a / views_a) * annual_traffic * val_per_conv
projected_annual_rev = current_annual_rev * (1 + avg_lift)
annual_gain = projected_annual_rev - current_annual_rev

# --- TOP LEVEL KPI DASHBOARD ---
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Win Confidence", f"{prob_b_better:.1%}")
    st.caption("Probability Variant > Control")

with kpi2:
    st.metric("Expected Lift", f"{avg_lift:+.1%}")
    st.caption("Most likely improvement")

with kpi3:
    color = "green" if prob_b_better > 0.95 else "orange" if prob_b_better > 0.80 else "red"
    rec = "DEPLOY" if prob_b_better > 0.95 else "WAIT"
    st.markdown(f"**Recommendation**")
    st.markdown(f"<h1 style='color:{color}; margin-top:-15px;'>{rec}</h1>", unsafe_allow_html=True)

st.divider()

# --- VISUALS ---
tab1, tab2 = st.tabs(["📊 Performance & Bayesian Evolution", "💰 Revenue Impact"])

with tab1:
    # --- NEW: BAYESIAN EVOLUTION (PDF CHART) ---
    st.subheader("The Bayesian Update: Prior vs. Current Data")
    st.write("How this experiment is shifting our historical baseline.")
    
    # Generate X-axis (range of CTRs to plot)
    x_max = max(sim_b.max(), sim_a.max(), hist_ctr) * 1.5
    x_range = np.linspace(0, x_max, 500)
    
    # Calculate PDF curves
    prior_pdf = stats.beta.pdf(x_range, alpha_p, beta_p)
    control_pdf = stats.beta.pdf(x_range, alpha_p + clicks_a, beta_p + (views_a - clicks_a))
    variant_pdf = stats.beta.pdf(x_range, alpha_p + clicks_b, beta_p + (views_b - clicks_b))
    
    fig_evol = go.Figure()
    fig_evol.add_trace(go.Scatter(x=x_range, y=prior_pdf, name="Historical Prior", 
                                 line=dict(color='black', dash='dash', width=2)))
    fig_evol.add_trace(go.Scatter(x=x_range, y=control_pdf, name="Control (Updated)", 
                                 fill='tozeroy', line_color='#95a5a6'))
    fig_evol.add_trace(go.Scatter(x=x_range, y=variant_pdf, name="Variant (Updated)", 
                                 fill='tozeroy', line_color='#2ecc71'))
    
    fig_evol.update_layout(xaxis_title="Conversion Rate (CTR)", yaxis_title="Density", height=400)
    st.plotly_chart(fig_evol, use_container_width=True)

    st.divider()

    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        st.subheader("Performance Range")
        control_mean, variant_mean = np.mean(sim_a), np.mean(sim_b)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=['Control', 'Variant'], x=[control_mean, variant_mean],
            orientation='h', marker_color=['#95a5a6', '#2ecc71'],
            text=[f"{control_mean:.2%}", f"{variant_mean:.2%}"], textposition='auto',
            error_x=dict(type='data', symmetric=False,
                array=[np.percentile(sim_b, 95)-variant_mean, np.percentile(sim_a, 95)-control_mean], # Error in order fix
                visible=True)
        ))
        fig_bar.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.subheader("Probability of Outcomes")
        big_win = (relative_lift > 0.10).mean()
        small_win = ((relative_lift <= 0.10) & (relative_lift > 0)).mean()
        loss = (relative_lift <= 0).mean()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Big Win (>10%)", "Small Win", "Loss"],
            values=[big_win, small_win, loss],
            hole=.4, marker_colors=['#2ecc71', '#82e0aa', '#e74c3c']
        )])
        fig_pie.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Annual Financial Projection")
    impact_data = {
        "Metric": ["Conversion Rate", "Annual Revenue", "Revenue Delta"],
        "Current (Baseline)": [f"{clicks_a/views_a:.2%}", f"${current_annual_rev:,.0f}", "-"],
        "Projected (Variant)": [f"{np.mean(sim_b):.2%}", f"${projected_annual_rev:,.0f}", f"+${annual_gain:,.0f}"]
    }
    st.table(pd.DataFrame(impact_data))
    st.info(f"**Insight:** Based on your history and current data, the Variant is expected to add **${annual_gain:,.0f}** to annual revenue.")
