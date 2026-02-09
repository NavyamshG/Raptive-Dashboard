import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab: Comparative Statistical Analysis
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")

# Custom CSS for punchier metrics
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    .description-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 CTR Inference Lab")

# Dashboard Description
st.markdown("""
<div class="description-box">
    <strong>Dashboard Objective:</strong> I'm trying to show how p-values and confidence vary when we use a Bernoulli or Binomial approach and run the data with T-test, Z-test, and Fisher's Exact values test.
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
    scenario = st.selectbox("Choose a scenario or Manual Entry:", 
                            ["Manual Entry", "Small Sample (Low Power)", "The Marginal Win", "Clear Winner"])
    
    presets = {
        "Manual Entry": (20, 200, 35, 200),
        "Small Sample (Low Power)": (2, 20, 5, 20),
        "The Marginal Win": (100, 1000, 125, 1000),
        "Clear Winner": (50, 1000, 120, 1000)
    }
    def_x1, def_n1, def_x2, def_n2 = presets[scenario]

    st.divider()
    st.header("🎛️ Experiment Inputs")
    colA, colB = st.columns(2)
    x1 = colA.number_input("Control Clicks", value=def_x1, min_value=0)
    n1 = colA.number_input("Control Views", value=def_n1, min_value=1)
    x2 = colB.number_input("Variant Clicks", value=def_x2, min_value=0)
    n2 = colB.number_input("Variant Views", value=def_n2, min_value=1)
    
    alpha = st.slider("Significance α (Type I Error)", 0.01, 0.20, 0.05)

    st.divider()
    with st.expander("🚨 Peeking Demo Settings"):
        show_peeking = st.toggle("Show Peeking Demo", value=True)
        looks = st.slider("Number of looks (k)", 2, 30, 15)
        sims = st.slider("Simulations", 100, 2000, 500)
        seed = st.number_input("Seed", value=42)

# -------------------------
# Processing
# -------------------------
validate_counts(int(x1), int(n1), "Control")
validate_counts(int(x2), int(n2), "Variant")

p1, p2 = x1/n1, x2/n2
diff = p2 - p1
rel = diff / p1 if p1 > 0 else np.nan

# Execute Tests
z_stat, z_p = two_prop_ztest(x1, n1, x2, n2)
t_stat, t_df, t_p = welch_ttest_bernoulli(x1, n1, x2, n2)
odds, f_p = fishers_exact(x1, n1, x2, n2)

# Verdict Logic
tests = {"z-test": z_p, "t-test": t_p, "Fisher": f_p}
wins = [name for name, p in tests.items() if np.isfinite(p) and p <= alpha]
losses = [name for name, p in tests.items() if np.isfinite(p) and p > alpha]

# CIs
wald_d, wald_lo, wald_hi = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
newc_d, newc_lo, newc_hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)

# -------------------------
# Main UI
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Control CTR", pct(p1))
c2.metric("Variant CTR", pct(p2))
lift_color = "normal" if (z_p <= alpha) else "off"
c3.metric("Δ CTR (Abs)", pct(diff), f"Rel: {pct(rel)}", delta_color=lift_color)
c4.metric("Confidence (z-test)", pct(1 - z_p) if np.isfinite(z_p) else "NA")

st.divider()

# 1) Visual Overlap & Test Verdicts
st.subheader("1) Comparative Statistical Verdicts")
col_plot, col_results = st.columns([1.5, 1])

with col_plot:
    x_axis = np.linspace(max(0, min(p1, p2) - 0.15), min(1, max(p1, p2) + 0.15), 500)
    se1 = math.sqrt(p1*(1-p1)/n1) if p1 > 0 else 0.01
    se2 = math.sqrt(p2*(1-p2)/n2) if p2 > 0 else 0.01
    y1, y2 = stats.norm.pdf(x_axis, p1, se1), stats.norm.pdf(x_axis, p2, se2)
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(x=x_axis, y=y1, fill='tozeroy', name='Control (Binomial Approx)', line_color='#636EFA'))
    fig_dist.add_trace(go.Scatter(x=x_axis, y=y2, fill='tozeroy', name='Variant (Binomial Approx)', line_color='#00CC96'))
    fig_dist.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="CTR Range", yaxis_title="Probability Density")
    st.plotly_chart(fig_dist, use_container_width=True)

with col_results:
    st.markdown("**Test Performance**")
    if wins:
        st.success(f"**Significant Win according to:** {', '.join(wins)}")
    if losses:
        st.error(f"**No Significance according to:** {', '.join(losses)}")

    fig_methods = go.Figure()
    fig_methods.add_trace(go.Bar(
        x=["z-test", "t-test", "Fisher"],
        y=[z_p if np.isfinite(z_p) else 1.0, t_p if np.isfinite(t_p) else 1.0, f_p],
        marker_color=['#00CC96' if p <= alpha else '#EF553B' for p in [z_p, t_p, f_p]],
        text=[f"{p:.4f}" for p in [z_p, t_p, f_p]],
        textposition="auto"
    ))
    fig_methods.add_hline(y=alpha, line_dash="dash", line_color="black")
    fig_methods.update_layout(yaxis_title="p-value", height=250, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig_methods, use_container_width=True)
    st.caption(f"Horizontal line represents α={alpha}")

st.divider()

# 2) Confidence Interval Comparison
st.subheader("2) Delta Confidence Intervals")
fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(x=[diff], y=["Wald (Binomial)"], mode="markers", error_x=dict(type="data", array=[wald_hi - diff], arrayminus=[diff - wald_lo], visible=True), marker=dict(size=12, color="#AB63FA")))
fig_ci.add_trace(go.Scatter(x=[diff], y=["Newcombe (Wilson Score)"], mode="markers", error_x=dict(type="data", array=[newc_hi - diff], arrayminus=[diff - newc_lo], visible=True), marker=dict(size=12, color="#EF553B")))
fig_ci.add_vline(x=0, line_dash="dash", line_color="gray")
fig_ci.update_layout(height=280, xaxis_title="Abs Difference (B - A)")
st.plotly_chart(fig_ci, use_container_width=True)

# 3) Peeking Danger Demo
if show_peeking:
    st.subheader("3) The Danger of Peeking (False Positive Risk)")
    rng = np.random.default_rng(int(seed))
    n_total = n1 + n2
    sample_points = np.linspace(10, n_total, looks).astype(int)
    
    # Simulating a null hypothesis journey
    example_p_journey = []
    a_hits, b_hits = rng.binomial(1, p1, size=n_total), rng.binomial(1, p1, size=n_total)
    
    for n_pt in sample_points:
        xa, xb = a_hits[:n_pt].sum(), b_hits[:n_pt].sum()
        _, p_val = two_prop_ztest(xa, n_pt, xb, n_pt)
        example_p_journey.append(p_val)

    # Simulation to calculate FPR
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
        st.info(f"By checking results {looks} times instead of once at the end, your error rate is {fp_rate/alpha:.1f}x higher than your alpha threshold.")
