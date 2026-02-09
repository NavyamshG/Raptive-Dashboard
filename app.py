import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab: Comparative Statistical Analysis
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")

# Custom CSS for high-impact headers and professional text blocks
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    
    .main-header { 
        font-size: 4.5rem; 
        font-weight: 900; 
        color: #1E1E1E; 
        margin-bottom: 0;
        line-height: 1.1;
    }
    
    .sub-header { 
        font-size: 1.4rem; 
        color: #5E5E5E; 
        margin-top: 5px; 
        margin-bottom: 25px; 
        font-weight: 400;
    }
    
    .description-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 25px;
    }
    
    .explanation-text {
        font-size: 1rem;
        line-height: 1.5;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Headers
st.markdown('<p class="main-header">📊 CTR Inference Lab</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Evaluating Estimator Robustness: How Distributional Assumptions Impact Significance in Discrete A/B Testing</p>', unsafe_allow_html=True)

# Dashboard Summary & Explanation
st.markdown("""
<div class="description-box">
    <strong>Project Objective:</strong> 
    Evaluating p-value and confidence interval variance across Bernoulli and Binomial modeling frameworks to compare the sensitivity of Z-test, Welch’s T-test, and Fisher’s Exact methods.
    <hr style="margin: 15px 0; border: 0; border-top: 1px solid #ddd;">
    <div class="explanation-text">
        <strong>Simple Explanation:</strong><br>
        This lab demonstrates how different mathematical "lenses" interpret the same data, comparing the standard bell-curve approximations of Z and T-tests against the exact probability calculations of Fisher’s method. It reveals whether a statistical "win" is a robust result or simply a byproduct of the specific distribution and test selected.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------
# Helpers
# -------------------------
def validate_counts(x: int, n: int, label: str) -> None:
    if n <= 0:
        st.error(f"{label}: trials must be > 0.")
        st.stop()
    if x < 0:
        st.error(f"{label}: clicks must be ≥ 0.")
        st.stop()
    if x > n:
        st.error(f"{label}: clicks must be ≤ trials.")
        st.stop()

def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0: return np.nan, np.nan
    z = (p2 - p1) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

def welch_ttest_bernoulli(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    if n1 <= 1 or n2 <= 1: return np.nan, np.nan, np.nan
    s1_sq = (n1 / (n1 - 1)) * p1 * (1 - p1)
    s2_sq = (n2 / (n2 - 1)) * p2 * (1 - p2)
    se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    if se == 0: return np.nan, np.nan, np.nan
    t = (p2 - p1) / se
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, df, p

def fishers_exact(x1, n1, x2, n2):
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    oddsratio, p = stats.fisher_exact(table, alternative="two-sided")
    return oddsratio, p

def wald_ci_diff(x1, n1, x2, n2, alpha=0.05):
    p1, p2 = x1 / n1, x2 / n2
    d = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return d, d - z * se, d + z * se

def wilson_ci_single(x, n, alpha=0.05):
    if n == 0: return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    return max(0.0, center - half), min(1.0, center + half)

def newcombe_ci_diff(x1, n1, x2, n2, alpha=0.05):
    l1, u1 = wilson_ci_single(x1, n1, alpha)
    l2, u2 = wilson_ci_single(x2, n2, alpha)
    d = (x2 / n2) - (x1 / n1)
    return d, (l2 - u1), (u2 - l1)

def pct(x):
    return f"{x:.2%}" if np.isfinite(x) else "NA"

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("🕹️ Scenario Selector")
    scenario = st.selectbox("Choose a scenario:", ["Manual Entry", "Small Sample", "Marginal Win", "Clear Winner"])
    presets = {"Manual Entry": (20, 200, 35, 200), "Small Sample": (2, 20, 5, 20), "Marginal Win": (100, 1000, 125, 1000), "Clear Winner": (50, 1000, 120, 1000)}
    def_x1, def_n1, def_x2, def_n2 = presets[scenario]
    
    st.divider()
    x1 = st.number_input("Control Clicks", value=def_x1, min_value=0)
    n1 = st.number_input("Control Views", value=def_n1, min_value=1)
    x2 = st.number_input("Variant Clicks", value=def_x2, min_value=0)
    n2 = st.number_input("Variant Views", value=def_n2, min_value=1)
    alpha = st.slider("Significance α", 0.01, 0.20, 0.05)

    st.divider()
    with st.expander("🚨 Peeking Simulation Settings"):
        show_peeking = st.toggle("Show Peeking Demo", value=True)
        looks = st.slider("Number of looks (k)", 2, 30, 15)
        sims = st.slider("Simulations", 100, 2000, 500)
        seed = st.number_input("Seed", value=42)

# -------------------------
# Calculations
# -------------------------
validate_counts(int(x1), int(n1), "Control")
validate_counts(int(x2), int(n2), "Variant")

p1, p2 = x1/n1, x2/n2
diff = p2 - p1
z_stat, z_p = two_prop_ztest(x1, n1, x2, n2)
t_stat, t_df, t_p = welch_ttest_bernoulli(x1, n1, x2, n2)
odds, f_p = fishers_exact(x1, n1, x2, n2)

tests = {"z-test": z_p, "t-test": t_p, "Fisher": f_p}
wins = [name for name, p in tests.items() if np.isfinite(p) and p <= alpha]
losses = [name for name, p in tests.items() if np.isfinite(p) and p > alpha]

wald_d, wald_lo, wald_hi = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
newc_d, newc_lo, newc_hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)

# -------------------------
# Main UI
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Control CTR", pct(p1))
c2.metric("Variant CTR", pct(p2))
c3.metric("Δ CTR (Abs)", pct(diff))
c4.metric("Confidence (Z)", pct(1 - z_p) if np.isfinite(z_p) else "NA")

st.divider()

# 1) Visual Overlap & Verdicts
st.subheader("1) Comparative Statistical Verdicts")
col_plot, col_results = st.columns([1.5, 1])

with col_plot:
    x_axis = np.linspace(max(0, min(p1, p2) - 0.15), min(1, max(p1, p2) + 0.15), 500)
    y1 = stats.norm.pdf(x_axis, p1, math.sqrt(p1*(1-p1)/n1))
    y2 = stats.norm.pdf(x_axis, p2, math.sqrt(p2*(1-p2)/n2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y1, fill='tozeroy', name='Control', line_color='#636EFA'))
    fig.add_trace(go.Scatter(x=x_axis, y=y2, fill='tozeroy', name='Variant', line_color='#00CC96'))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_results:
    if wins:
        st.success(f"**Winner according to:** {', '.join(wins)}")
    if losses:
        st.error(f"**No Significance according to:** {', '.join(losses)}")
    
    st.markdown("**p-value Comparison**")
    fig_p = go.Figure(go.Bar(
        x=list(tests.keys()), 
        y=list(tests.values()),
        marker_color=['#00CC96' if p <= alpha else '#EF553B' for p in tests.values()],
        text=[f"{p:.4f}" for p in tests.values()],
        textposition='auto'
    ))
    fig_p.add_hline(y=alpha, line_dash="dash")
    fig_p.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_p, use_container_width=True)

# Magic Scenario Callout placed under the graphs
    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; border: 1px solid #add8e6; margin-top: 20px;">
        <span style="font-size: 1.1rem;">✨ <strong>To see a magic:</strong> 
        Set Control clicks: <strong>20</strong>, Control views: <strong>200</strong>, 
        Variant clicks: <strong>35</strong>, Variant views: <strong>200</strong>, 
        and Significance level: <strong>0.04</strong></span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# 2) Confidence Intervals
st.subheader("2) Delta Confidence Intervals")
st.markdown(f"**Wald Interval (Binomial):** `{pct(wald_lo)}` to `{pct(wald_hi)}`  \n"
            f"**Newcombe Interval (Wilson):** `{pct(newc_lo)}` to `{pct(newc_hi)}`")

fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(x=[diff], y=["Wald (Binomial)"], mode="markers", error_x=dict(type="data", array=[wald_hi - diff], arrayminus=[diff - wald_lo], visible=True), marker=dict(size=12, color="#AB63FA")))
fig_ci.add_trace(go.Scatter(x=[diff], y=["Newcombe (Wilson)"], mode="markers", error_x=dict(type="data", array=[newc_hi - diff], arrayminus=[diff - newc_lo], visible=True), marker=dict(size=12, color="#EF553B")))
fig_ci.add_vline(x=0, line_dash="dash", line_color="gray")
fig_ci.update_layout(height=300, xaxis_title="Abs Difference", showlegend=False)
st.plotly_chart(fig_ci, use_container_width=True)

st.divider()

# 3) Peeking Danger Demo (REINSTATED)
if show_peeking:
    st.subheader("3) The Danger of Peeking (False Positive Risk)")
    rng = np.random.default_rng(int(seed))
    n_total = n1 + n2
    sample_points = np.linspace(10, n_total, looks).astype(int)
    
    example_p_journey = []
    a_hits, b_hits = rng.binomial(1, p1, size=n_total), rng.binomial(1, p1, size=n_total)
    for n_pt in sample_points:
        xa, xb = a_hits[:n_pt].sum(), b_hits[:n_pt].sum()
        _, p_val = two_prop_ztest(xa, n_pt, xb, n_pt)
        example_p_journey.append(p_val)

    any_fp = 0
    for _ in range(int(sims)):
        a, b = rng.binomial(1, p1, size=n_total), rng.binomial(1, p1, size=n_total)
        for n_pt in sample_points:
            xa, xb = a[:n_pt].sum(), b[:n_pt].sum()
            _, p = two_prop_ztest(xa, n_pt, xb, n_pt)
            if p <= alpha:
                any_fp += 1
                break
    fp_rate = any_fp / sims
    
    col_peek_chart, col_peek_stats = st.columns([2, 1])
    with col_peek_chart:
        fig_journey = go.Figure(go.Scatter(x=sample_points, y=example_p_journey, mode='lines+markers', line_color='#FF4B4B'))
        fig_journey.add_hline(y=alpha, line_dash="dash")
        fig_journey.update_layout(height=350, yaxis_title="P-Value Over Time", yaxis_range=[0, 1])
        st.plotly_chart(fig_journey, use_container_width=True)
    with col_peek_stats:
        st.metric("Actual False Positive Rate", f"{fp_rate:.1%}")
        st.info(f"Checking results {looks} times instead of once increases your error rate by {fp_rate/alpha:.1f}x.")
