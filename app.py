import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab (Simple + Intuitive)
# - Keep: z-test + t-test + Fisher comparison
# - Add: Simple CI visual (Wald vs Newcombe/Wilson) for Δ CTR
# - Keep: Simple peeking demo
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")
st.title("📊 CTR Inference Lab")
st.caption("Compare z-test vs t-test vs Fisher for CTR. See how CI methods differ. Plus: peeking inflates false positives.")

with st.expander("📘 What this demonstrates", expanded=True):
    st.markdown(
        """
**Three simple takeaways:**
1) **Same CTR data, different tests → different p-values** (especially at small n / low CTR).
2) **Confidence intervals (CI) can differ by method** (Wald can be unstable; Wilson/Newcombe is usually safer).
3) **Peeking** (checking results many times and stopping early) **inflates false positives** even when there is no real effect.
        """
    )

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
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p2 - p1) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

def welch_ttest_bernoulli(x1, n1, x2, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    if n1 <= 1 or n2 <= 1:
        return np.nan, np.nan, np.nan

    s1_sq = (n1 / (n1 - 1)) * p1 * (1 - p1)
    s2_sq = (n2 / (n2 - 1)) * p2 * (1 - p2)

    se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    if se == 0:
        return np.nan, np.nan, np.nan

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

def norm_ppf(p: float) -> float:
    return stats.norm.ppf(p)

def wald_ci_diff(x1, n1, x2, n2, alpha=0.05):
    p1 = x1 / n1
    p2 = x2 / n2
    d = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = norm_ppf(1 - alpha / 2)
    return d, d - z * se, d + z * se

def wilson_ci_single(x, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    z = norm_ppf(1 - alpha / 2)
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
    st.header("🎛️ Inputs")

    with st.expander("A/B counts", expanded=True):
        col1, col2 = st.columns(2)
        x1 = col1.number_input("Control clicks", value=20, min_value=0, step=1)
        n1 = col1.number_input("Control views", value=200, min_value=1, step=1)
        x2 = col2.number_input("Variant clicks", value=30, min_value=0, step=1)
        n2 = col2.number_input("Variant views", value=200, min_value=1, step=1)

    alpha = st.slider("Significance α", 0.01, 0.20, 0.05, 0.01)

    with st.expander("🚨 Peeking demo (optional)", expanded=False):
        show_peeking = st.toggle("Show peeking demo", value=True)
        looks = st.slider("Number of looks (k)", 2, 30, 10, 1)
        final_n = st.slider("Final sample size per arm", 200, 50000, 5000, 100)
        base_p = st.slider("True CTR under H0", 0.001, 0.20, 0.05, 0.001)
        sims = st.slider("Simulations", 200, 3000, 1000, 100)
        seed = st.number_input("Seed", value=7, min_value=0, step=1)

# -------------------------
# Validate
# -------------------------
validate_counts(int(x1), int(n1), "Control")
validate_counts(int(x2), int(n2), "Variant")
x1, n1, x2, n2 = int(x1), int(n1), int(x2), int(n2)

p1 = x1 / n1
p2 = x2 / n2
diff = p2 - p1
rel = diff / p1 if p1 > 0 else np.nan

# -------------------------
# Tests + CIs
# -------------------------
z_stat, z_p = two_prop_ztest(x1, n1, x2, n2)
t_stat, t_df, t_p = welch_ttest_bernoulli(x1, n1, x2, n2)
odds, f_p = fishers_exact(x1, n1, x2, n2)

wald_d, wald_lo, wald_hi = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
newc_d, newc_lo, newc_hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)

# -------------------------
# KPIs
# -------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Control CTR", pct(p1), f"{x1}/{n1}")
with c2:
    st.metric("Variant CTR", pct(p2), f"{x2}/{n2}")
with c3:
    st.metric("Δ CTR", pct(diff), f"Rel: {pct(rel) if np.isfinite(rel) else 'NA'}")
with c4:
    st.metric("α", f"{alpha:.2f}")

st.divider()

# -------------------------
# 1) Test comparison (z, t, Fisher)
# -------------------------
st.subheader("1) Same data, different tests")

colL, colR = st.columns([1.2, 1])
with colL:
    st.markdown(
        """
**Goal:** test whether CTR differs between A and B.

- **z-test:** normal approximation for proportions  
- **t-test:** treats 0/1 as numeric (often close for big n)  
- **Fisher:** exact test for the 2×2 table (best when counts are small)
        """
    )
with colR:
    st.markdown("**p-values**")
    st.metric("z-test", "NA" if not np.isfinite(z_p) else f"{z_p:.4f}", "✅ sig" if (np.isfinite(z_p) and z_p <= alpha) else "—")
    st.metric("t-test", "NA" if not np.isfinite(t_p) else f"{t_p:.4f}", "✅ sig" if (np.isfinite(t_p) and t_p <= alpha) else "—")
    st.metric("Fisher", f"{f_p:.4f}", "✅ sig" if f_p <= alpha else "—")

fig_methods = go.Figure()
fig_methods.add_trace(go.Bar(
    x=["z-test", "t-test", "Fisher"],
    y=[
        z_p if np.isfinite(z_p) else 1.0,
        t_p if np.isfinite(t_p) else 1.0,
        f_p
    ],
    text=[
        f"{z_p:.4f}" if np.isfinite(z_p) else "NA",
        f"{t_p:.4f}" if np.isfinite(t_p) else "NA",
        f"{f_p:.4f}"
    ],
    textposition="auto"
))
fig_methods.add_hline(y=alpha)
fig_methods.update_layout(yaxis_title="p-value", height=320, showlegend=False)
st.plotly_chart(fig_methods, use_container_width=True)
st.caption("Bars below the α line are statistically significant.")

st.divider()

# -------------------------
# 2) Simple CI visual (Wald vs Newcombe/Wilson)
# -------------------------
st.subheader("2) Confidence interval for Δ CTR (two methods)")

st.markdown(
    """
**Same estimate (Δ CTR), different CI math.**  
Wald can be too optimistic or weird at small n / extreme CTR. Newcombe/Wilson is usually more stable.
    """
)

fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(
    x=[wald_lo, wald_hi],
    y=["Wald", "Wald"],
    mode="lines",
    name="Wald CI"
))
fig_ci.add_trace(go.Scatter(
    x=[newc_lo, newc_hi],
    y=["Newcombe/Wilson", "Newcombe/Wilson"],
    mode="lines",
    name="Newcombe/Wilson CI"
))
fig_ci.add_trace(go.Scatter(
    x=[diff],
    y=["Observed Δ", "Observed Δ"],
    mode="markers",
    name="Observed Δ",
    marker=dict(size=10)
))
fig_ci.add_vline(x=0)

fig_ci.update_layout(
    xaxis_title="Δ CTR (B − A)",
    yaxis_title="",
    height=320,
    showlegend=True
)
st.plotly_chart(fig_ci, use_container_width=True)

cA, cB = st.columns(2)
with cA:
    st.metric("Wald CI", f"[{pct(wald_lo)}, {pct(wald_hi)}]")
with cB:
    st.metric("Newcombe/Wilson CI", f"[{pct(newc_lo)}, {pct(newc_hi)}]")

st.caption("If the CI crosses 0, the data is compatible with no difference (at this confidence level).")

st.divider()

# -------------------------
# 3) Peeking demo
# -------------------------
if show_peeking:
    st.subheader("3) Peeking inflates false positives (simple demo)")

    st.markdown(
        """
**Setup:** there is **no real difference** (A and B have the same true CTR).  
But if you check results many times and stop when p ≤ α, you will “find” a win more often than α.
        """
    )

    rng = np.random.default_rng(int(seed))
    look_ns = np.unique(np.round(np.linspace(final_n / looks, final_n, looks)).astype(int))

    any_fp = 0
    first_hit = []

    for _ in range(int(sims)):
        a = rng.binomial(1, base_p, size=final_n)
        b = rng.binomial(1, base_p, size=final_n)

        hit_at = None
        for i, nlook in enumerate(look_ns, start=1):
            xa = int(a[:nlook].sum())
            xb = int(b[:nlook].sum())
            _, p = two_prop_ztest(xa, nlook, xb, nlook)
            if np.isfinite(p) and p <= alpha:
                hit_at = i
                break

        if hit_at is not None:
            any_fp += 1
            first_hit.append(hit_at)

    fp_rate = any_fp / sims
    approx = 1 - (1 - alpha) ** looks

    pA, pB, pC = st.columns(3)
    with pA:
        st.metric("False positives (with peeking)", f"{fp_rate:.1%}")
    with pB:
        st.metric("Target (no peeking)", f"{alpha:.1%}")
    with pC:
        st.metric("Rough intuition", f"{approx:.1%}")

    fig_peek = go.Figure()
    if len(first_hit) > 0:
        fig_peek.add_trace(go.Histogram(x=first_hit, nbinsx=min(looks, 20)))
    fig_peek.update_layout(
        xaxis_title="Which look triggered significance first (1..k)",
        yaxis_title="count",
        height=320
    )
    st.plotly_chart(fig_peek, use_container_width=True)

    st.caption("This is why sequential testing needs correction (alpha-spending, group sequential tests, always-valid methods).")
