import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab (Full Merged Version)
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")

# Custom CSS for punchier metrics
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 CTR Inference Lab")

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
    st.header("🎛️ Experiment Inputs")
    colA, colB = st.columns(2)
    x1 = colA.number_input("Control Clicks", value=20, min_value=0)
    n1 = colA.number_input("Control Views", value=200, min_value=1)
    x2 = colB.number_input("Variant Clicks", value=35, min_value=0)
    n2 = colB.number_input("Variant Views", value=200, min_value=1)
    
    alpha = st.slider("Significance α", 0.01, 0.20, 0.05)

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

# Tests
z_stat, z_p = two_prop_ztest(x1, n1, x2, n2)
t_stat, t_df, t_p = welch_ttest_bernoulli(x1, n1, x2, n2)
odds, f_p = fishers_exact(x1, n1, x2, n2)
is_significant = z_p <= alpha

# CIs
wald_d, wald_lo, wald_hi = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
newc_d, newc_lo, newc_hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)

# -------------------------
# KPI Section
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Control CTR", pct(p1))
c2.metric("Variant CTR", pct(p2))

# Color coding for the lift
lift_color = "normal" if is_significant else "off"
c3.metric("Δ CTR (Abs)", pct(diff), f"Rel: {pct(rel)}", delta_color=lift_color)
c4.metric("Statistical Confidence", pct(1 - z_p) if np.isfinite(z_p) else "NA")

st.divider()

# -------------------------
# 1) Visual Evidence (The Overlap)
# -------------------------
st.subheader("1) The 'Overlap' View")
col_plot, col_text = st.columns([2, 1])

with col_plot:
    # Distribution Curves
    x_axis = np.linspace(max(0, min(p1, p2) - 0.15), min(1, max(p1, p2) + 0.15), 500)
    se1 = math.sqrt(p1*(1-p1)/n1) if p1 > 0 else 0.01
    se2 = math.sqrt(p2*(1-p2)/n2) if p2 > 0 else 0.01
    
    y1 = stats.norm.pdf(x_axis, p1, se1)
    y2 = stats.norm.pdf(x_axis, p2, se2)
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(x=x_axis, y=y1, fill='tozeroy', name='Control', line_color='#636EFA'))
    fig_dist.add_trace(go.Scatter(x=x_axis, y=y2, fill='tozeroy', name='Variant', line_color='#00CC96'))
    fig_dist.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="CTR Range", yaxis_title="Probability Density")
    st.plotly_chart(fig_dist, use_container_width=True)

with col_text:
    st.markdown("**Verdict**")
    if is_significant:
        st.success(f"**Significant Win!** 🎉\n\nVariant B outperformed Control A. We are {(1-z_p):.1%} confident this isn't just luck.")
    else:
        st.warning(f"**Not Significant** 😴\n\nThe overlap is too high. Current p-value: {z_p:.4f}. You likely need more samples.")
    
    st.markdown("**Test Comparison (p-values)**")
    st.caption("z-test")
    st.progress(min(z_p, 1.0) if np.isfinite(z_p) else 1.0)
    st.caption("Fisher Exact")
    st.progress(min(f_p, 1.0))

st.divider()

# -------------------------
# 2) CI Comparison
# -------------------------
st.subheader("2) Confidence Intervals for Δ CTR")
st.markdown("Comparing the standard **Wald** interval against the more robust **Newcombe/Wilson** method.")

fig_ci = go.Figure()
# Wald
fig_ci.add_trace(go.Scatter(
    x=[diff], y=["Wald"], mode="markers", name="Wald",
    error_x=dict(type="data", symmetric=False, array=[wald_hi - diff], arrayminus=[diff - wald_lo], visible=True),
    marker=dict(size=12, color="#AB63FA")
))
# Newcombe
fig_ci.add_trace(go.Scatter(
    x=[diff], y=["Newcombe/Wilson"], mode="markers", name="Newcombe",
    error_x=dict(type="data", symmetric=False, array=[newc_hi - diff], arrayminus=[diff - newc_lo], visible=True),
    marker=dict(size=12, color="#EF553B")
))
fig_ci.add_vline(x=0, line_dash="dash", line_color="gray")
fig_ci.update_layout(height=300, xaxis_title="Estimated Difference (B - A)", showlegend=False)
st.plotly_chart(fig_ci, use_container_width=True)

ca, cb = st.columns(2)
ca.info(f"**Wald CI:** [{pct(wald_lo)}, {pct(wald_hi)}]")
cb.info(f"**Newcombe CI:** [{pct(newc_lo)}, {pct(newc_hi)}]")

st.divider()

# -------------------------
# 3) Peeking Demo
# -------------------------
if show_peeking:
    st.subheader("3) The Danger of Peeking")
    st.markdown("Even if **A and B are identical**, checking your results repeatedly increases the chance of a False Positive.")
    
    # Simulation: P-value Journey
    rng = np.random.default_rng(int(seed))
    # We simulate a "Live Experiment" where true CTR is equal to p1 for both
    n_total = n1 + n2
    sample_points = np.linspace(10, n_total, looks).astype(int)
    
    any_fp = 0
    # One example journey to plot
    example_p_journey = []
    a_hits = rng.binomial(1, p1, size=n_total)
    b_hits = rng.binomial(1, p1, size=n_total)
    
    for n_pt in sample_points:
        xa, xb = a_hits[:n_pt].sum(), b_hits[:n_pt].sum()
        _, p_val = two_prop_ztest(xa, n_pt, xb, n_pt)
        example_p_journey.append(p_val)

    # Multi-simulation False Positive Rate
    for _ in range(int(sims)):
        a = rng.binomial(1, p1, size=n_total)
        b = rng.binomial(1, p1, size=n_total)
        for n_pt in sample_points:
            xa, xb = a[:n_pt].sum(), b[:n_pt].sum()
            _, p = two_prop_ztest(xa, n_pt, xb, n_pt)
            if p <= alpha:
                any_fp += 1
                break

    fp_rate = any_fp / sims
    
    col_peek_chart, col_peek_stats = st.columns([2, 1])
    
    with col_peek_chart:
        fig_journey = go.Figure()
        fig_journey.add_trace(go.Scatter(x=sample_points, y=example_p_journey, mode='lines+markers', name='P-value Journey', line_color='#FF4B4B'))
        fig_journey.add_hline(y=alpha, line_dash="dash", line_color="black", annotation_text="Significance Threshold")
        fig_journey.update_layout(height=350, xaxis_title="Sample Size (Time)", yaxis_title="P-Value", yaxis_range=[0, 1])
        st.plotly_chart(fig_journey, use_container_width=True)
        st.caption("A single simulation showing how the p-value 'dances' around the threshold.")

    with col_peek_stats:
        st.metric("Actual False Positive Rate", f"{fp_rate:.1%}")
        st.metric("Target Error Rate (α)", f"{alpha:.1%}")
        st.error(f"By peeking {looks} times, your risk of a false win increased by {(fp_rate/alpha if alpha>0 else 0):.1f}x!")
