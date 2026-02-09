import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab: The "Executive" Edition
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")

# Custom CSS for UI polish
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    .main-summary {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #636EFA;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 CTR Inference Lab")

# -------------------------
# Helpers
# -------------------------
def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0: return np.nan, np.nan
    z = (p2 - p1) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

def fishers_exact(x1, n1, x2, n2):
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    _, p = stats.fisher_exact(table, alternative="two-sided")
    return p

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

def newcombe_ci_diff(x1, n1, x2, n2, alpha=0.05):
    def wilson_ci(x, n):
        if n == 0: return 0, 0
        z = stats.norm.ppf(1 - alpha / 2)
        p = x / n
        denom = 1 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
        return max(0.0, center - half), min(1.0, center + half)
    l1, u1 = wilson_ci(x1, n1)
    l2, u2 = wilson_ci(x2, n2)
    d = (x2 / n2) - (x1 / n1)
    return d, (l2 - u1), (u2 - l1)

def pct(x):
    return f"{x:.2%}" if np.isfinite(x) else "NA"

# -------------------------
# 1) Scenario Selector (Add-on)
# -------------------------
with st.sidebar:
    st.header("🕹️ Presets & Scenarios")
    scenario = st.selectbox("Choose a common scenario:", 
                            ["Manual Entry", "The Marginal Win", "Small Sample Chaos", "Clear Winner", "Equal Performance"])
    
    # Preset Values
    presets = {
        "Manual Entry": (20, 200, 35, 200),
        "The Marginal Win": (100, 1000, 125, 1000),
        "Small Sample Chaos": (2, 20, 5, 20),
        "Clear Winner": (50, 1000, 120, 1000),
        "Equal Performance": (50, 500, 52, 500)
    }
    
    default_vals = presets[scenario]

    st.header("🎛️ Experiment Inputs")
    colA, colB = st.columns(2)
    x1 = colA.number_input("Control Clicks", value=default_vals[0], min_value=0)
    n1 = colA.number_input("Control Views", value=default_vals[1], min_value=1)
    x2 = colB.number_input("Variant Clicks", value=default_vals[2], min_value=0)
    n2 = colB.number_input("Variant Views", value=default_vals[3], min_value=1)
    
    alpha = st.slider("Significance α", 0.01, 0.20, 0.05)

    with st.expander("🚨 Peeking Demo Settings"):
        show_peeking = st.toggle("Show Peeking Demo", value=True)
        looks = st.slider("Number of looks (k)", 2, 30, 15)
        sims = st.slider("Simulations", 100, 2000, 500)

# -------------------------
# Processing
# -------------------------
p1, p2 = x1/n1, x2/n2
diff = p2 - p1
rel = diff / p1 if p1 > 0 else np.nan

z_stat, z_p = two_prop_ztest(x1, n1, x2, n2)
f_p = fishers_exact(x1, n1, x2, n2)
t_stat, t_df, t_p = welch_ttest_bernoulli(x1, n1, x2, n2)

tests = {"z-test": z_p, "t-test": t_p, "Fisher": f_p}
wins = [name for name, p in tests.items() if p <= alpha]
losses = [name for name, p in tests.items() if p > alpha]
is_sig = z_p <= alpha

# -------------------------
# 4) Executive Summary (Add-on)
# -------------------------
summary_container = st.container()
with summary_container:
    st.markdown('<div class="main-summary">', unsafe_allow_html=True)
    st.subheader("📝 Executive Summary")
    
    if is_sig and diff > 0:
        st.markdown(f"**Verdict:** 🚀 **Variant B is a clear winner.** It produced a **{pct(rel)}** relative lift over Control. With a confidence level of {(1-z_p):.1%}, we recommend full deployment.")
    elif is_sig and diff < 0:
        st.markdown(f"**Verdict:** ⚠️ **Control A is performing better.** Variant B decreased CTR by **{pct(abs(rel))}**. We recommend sticking with the Control.")
    else:
        st.markdown(f"**Verdict:** 😴 **Inconclusive.** We observed a {pct(diff)} absolute difference, but the data is currently too noisy to rule out random chance. Consider running the test longer.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Metrics
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Control CTR", pct(p1))
c2.metric("Variant CTR", pct(p2))
c3.metric("Δ CTR (Abs)", pct(diff), f"Rel: {pct(rel)}", delta_color="normal" if is_sig else "off")
c4.metric("Confidence (z-test)", pct(1 - z_p))

st.divider()

# -------------------------
# Distribution & Test Comparison
# -------------------------
col_plot, col_results = st.columns([1.5, 1])
with col_plot:
    st.subheader("1) The 'Overlap' View")
    x_axis = np.linspace(max(0, min(p1, p2) - 0.15), min(1, max(p1, p2) + 0.15), 500)
    se1 = math.sqrt(p1*(1-p1)/n1) if p1 > 0 else 0.01
    se2 = math.sqrt(p2*(1-p2)/n2) if p2 > 0 else 0.01
    y1, y2 = stats.norm.pdf(x_axis, p1, se1), stats.norm.pdf(x_axis, p2, se2)
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(x=x_axis, y=y1, fill='tozeroy', name='Control', line_color='#636EFA'))
    fig_dist.add_trace(go.Scatter(x=x_axis, y=y2, fill='tozeroy', name='Variant', line_color='#00CC96'))
    fig_dist.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_dist, use_container_width=True)

with col_results:
    st.subheader("Test Verdicts")
    if wins: st.success(f"**Won according to:** {', '.join(wins)}")
    if losses: st.error(f"**Lost according to:** {', '.join(losses)}")
    
    # Bar Chart
    fig_methods = go.Figure()
    fig_methods.add_trace(go.Bar(
        x=["z-test", "t-test", "Fisher"],
        y=[z_p, t_p, f_p],
        marker_color=['#00CC96' if p <= alpha else '#EF553B' for p in [z_p, t_p, f_p]],
        text=[f"{p:.4f}" for p in [z_p, t_p, f_p]], textposition="auto"
    ))
    fig_methods.add_hline(y=alpha, line_dash="dash")
    fig_methods.update_layout(height=220, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig_methods, use_container_width=True)

# -------------------------
# 5) MDE Pro-Tip (Add-on)
# -------------------------
if not is_sig:
    st.info("💡 **Pro-Tip: Why didn't we win?**")
    # Simple calculation for required p2 given current n and alpha
    z_crit = stats.norm.ppf(1 - alpha/2)
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    p2_needed = p1 + (z_crit * se_pool)
    st.write(f"To reach significance at your current sample size, Variant B would have needed a CTR of at least **{pct(p2_needed)}** (currently {pct(p2)}).")

st.divider()

# -------------------------
# CI & Peeking (Original Logic Kept)
# -------------------------
st.subheader("2) Confidence Intervals & Peeking")
ci_col, peek_col = st.columns([1, 1])

with ci_col:
    d, lo, hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(x=[d], y=["Newcombe"], error_x=dict(type="data", array=[hi-d], arrayminus=[d-lo], visible=True), mode="markers", marker=dict(size=12)))
    fig_ci.add_vline(x=0, line_dash="dash")
    fig_ci.update_layout(height=250, xaxis_title="Δ CTR (B-A)")
    st.plotly_chart(fig_ci, use_container_width=True)
    st.write(f"**Newcombe CI:** [{pct(lo)}, {pct(hi)}]")

with peek_col:
    if show_peeking:
        rng = np.random.default_rng(42)
        n_total = n1 + n2
        sample_points = np.linspace(10, n_total, looks).astype(int)
        example_p = []
        a_h, b_h = rng.binomial(1, p1, size=n_total), rng.binomial(1, p1, size=n_total)
        for n_pt in sample_points:
            _, p = two_prop_ztest(a_h[:n_pt].sum(), n_pt, b_h[:n_pt].sum(), n_pt)
            example_p.append(p)
        
        fig_peek = go.Figure(go.Scatter(x=sample_points, y=example_p, mode='lines+markers', line_color='#FF4B4B'))
        fig_peek.add_hline(y=alpha, line_dash="dash")
        fig_peek.update_layout(height=250, yaxis_range=[0,1], margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_peek, use_container_width=True)

if is_sig:
    st.balloons()
