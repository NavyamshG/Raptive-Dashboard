import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Raptive Decision Engine", layout="wide")

st.title("🏆 Executive A/B Decision Engine")
st.caption(f"Strategy & ROI Dashboard | Generated: Feb 2026")

# --- SIDEBAR: BUSINESS DATA ---
with st.sidebar:
    st.header("📊 Experiment Data")
    with st.expander("Conversion Data", expanded=True):
        col_a, col_b = st.columns(2)
        clicks_a = col_a.number_input("Control Clicks", value=100)
        views_a = col_a.number_input("Control Views", value=1000)
        clicks_b = col_b.number_input("Variant Clicks", value=130)
        views_b = col_b.number_input("Variant Views", value=1100)
    
    with st.expander("💰 Financials", expanded=True):
        val_per_conv = st.number_input("Value per Click ($)", value=50.0)
        monthly_traffic = st.number_input("Monthly Visitors", value=100000)

# --- BAYESIAN ENGINE ---
# Monte Carlo Simulation
sim_a = np.random.beta(clicks_a + 1, views_a - clicks_a + 1, 10000)
sim_b = np.random.beta(clicks_b + 1, views_b - clicks_b + 1, 10000)

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
    st.metric("Win Confidence", f"{prob_b_better:.1%}", 
              help="The probability that the Variant is genuinely better.")
    st.caption("How sure are we?")

with kpi2:
    st.metric("Expected Lift", f"{avg_lift:+.1%}", 
              help="The most likely percentage improvement.")
    st.caption("How much bigger?")

with kpi3:
    color = "green" if prob_b_better > 0.95 else "orange" if prob_b_better > 0.80 else "red"
    rec = "DEPLOY" if prob_b_better > 0.95 else "WAIT"
    st.markdown(f"**Recommendation**")
    st.markdown(f"<h1 style='color:{color}; margin-top:-15px;'>{rec}</h1>", unsafe_allow_html=True)

st.divider()

# --- SIMPLIFIED VISUALS ---
tab1, tab2 = st.tabs(["📊 Performance & Risk", "💰 Revenue Impact Table"])

with tab1:
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        st.subheader("Performance Comparison")
        # Simplified Bar Chart with Uncertainty
        control_mean = np.mean(sim_a)
        variant_mean = np.mean(sim_b)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=['Control', 'Variant'],
            x=[control_mean, variant_mean],
            orientation='h',
            marker_color=['#95a5a6', '#2ecc71'],
            text=[f"{control_mean:.2%}", f"{variant_mean:.2%}"],
            textposition='auto',
            # Error bars show the 90% confidence interval spread
            error_x=dict(
                type='data', 
                symmetric=False,
                array=[np.percentile(sim_a, 95)-control_mean, np.percentile(sim_b, 95)-variant_mean],
                arrayminus=[control_mean-np.percentile(sim_a, 5), variant_mean-np.percentile(sim_b, 5)],
                visible=True)
        ))
        fig.update_layout(xaxis_title="Conversion Rate (CTR)", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("Probability of Outcomes")
        # Bucketing the 10k simulations into 3 business scenarios
        big_win = (relative_lift > 0.10).mean()
        small_win = ((relative_lift <= 0.10) & (relative_lift > 0)).mean()
        loss = (relative_lift <= 0).mean()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Significant Win (>10%)", "Small Win (0-10%)", "Performance Loss"],
            values=[big_win, small_win, loss],
            hole=.4,
            marker_colors=['#2ecc71', '#82e0aa', '#e74c3c']
        )])
        fig_pie.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Estimated Annual Financial Impact")
    st.write("Projected revenue based on current conversion rates vs. the new Variant.")
    
    impact_data = {
        "Metric": ["Conversion Rate", "Annual Revenue", "Revenue Delta"],
        "Current (Control)": [f"{clicks_a/views_a:.2%}", f"${current_annual_rev:,.0f}", "-"],
        "Projected (Variant)": [f"{variant_mean:.2%}", f"${projected_annual_rev:,.0f}", f"+${annual_gain:,.0f}"]
    }
    df_impact = pd.DataFrame(impact_data)
    
    st.table(df_impact)
    
    st.info(f"**Bottom Line:** By switching to the Variant, we expect an annual revenue increase of **${annual_gain:,.2f}** with a **{prob_b_better:.1%}** confidence level.")

# --- FOOTER SUMMARY ---
if prob_b_better > 0.95:
    st.success(f"✅ **Strong Signal:** The Variant is consistently outperforming the Control. Deploying now is a high-probability win.")
else:
    st.warning(f"⚠️ **Wait:** We only have {prob_b_better:.1%} confidence. To reach the 95% 'Deploy' threshold, we need more impressions.")
