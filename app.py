import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab
# Compare inference methods for Bernoulli/Binomial outcomes:
# - Two-proportion z-test (pooled, H0: p1=p2)
# - Welch t-test on 0/1 outcomes
# - Fisher's exact test (exact for 2x2 table)
# CIs:
# - Wald CI for difference in proportions (simple, can misbehave)
# - Newcombe CI for difference (Wilson score intervals)
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")
st.title("📊 CTR Inference Lab")
st.caption("Compare z-test vs t-test vs Fisher for CTR (Bernoulli/Binomial), plus CI methods (Wald vs Newcombe/Wilson).")

with st.expander("📘 What this demonstrates", expanded=True):
    st.markdown(
        """
**Data model:** CTR is a Bernoulli rate (click/no-click), aggregated as Binomial counts.

**Why methods differ:**
- **Two-proportion z-test** uses a normal approximation (best at large n).
- **Welch t-test on 0/1 data** treats Bernoulli as continuous; often similar at large n, can differ at small n.
- **Fisher’s exact test** is exact for small counts (no normal approximation).

**Confidence intervals:**
- **Wald CI** (naive) can be inaccurate for small n or extreme CTRs.
- **Newcombe CI** (Wilson score based) is usually more reliable.
        """
    )

# -------------------------
# Utilities
# -------------------------
def validate_counts(x: int, n: int, label: str) -> None:
    if n <= 0:
        st.error(f"{label}: trials must be > 0.")
        st.stop()
    if x < 0:
        st.error(f"{label}: successes must be ≥ 0.")
        st.stop()
    if x > n:
        st.error(f"{label}: successes must be ≤ trials.")
        st.stop()

def norm_ppf(p: float) -> float:
    return stats.norm.ppf(p)

def two_prop_ztest(x1, n1, x2, n2, alternative="two-sided"):
    """
    Two-proportion z-test with pooled SE (classic for H0: p1=p2).
    Returns z, p_value, and SE_pooled.
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan, se
    z = (p2 - p1) / se  # Variant - Control

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":  # p2 > p1
        p = 1 - stats.norm.cdf(z)
    else:  # "less" p2 < p1
        p = stats.norm.cdf(z)
    return z, p, se

def welch_ttest_bernoulli(x1, n1, x2, n2, alternative="two-sided"):
    """
    Welch t-test on 0/1 outcomes.
    mean = p, var = p(1-p). Uses sample variances with ddof=1.
    Returns t, df, p_value, SE.
    """
    p1 = x1 / n1
    p2 = x2 / n2

    # sample variance for Bernoulli with ddof=1:
    # s^2 = (n/(n-1)) * p(1-p)  when data is 0/1 and p is sample mean
    if n1 <= 1 or n2 <= 1:
        return np.nan, np.nan, np.nan, np.nan

    s1_sq = (n1 / (n1 - 1)) * p1 * (1 - p1)
    s2_sq = (n2 / (n2 - 1)) * p2 * (1 - p2)

    se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    if se == 0:
        return np.nan, np.nan, np.nan, se

    t = (p2 - p1) / se

    # Welch-Satterthwaite df
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else np.nan

    if alternative == "two-sided":
        p = 2 * (1 - stats.t.cdf(abs(t), df))
    elif alternative == "greater":  # p2 > p1
        p = 1 - stats.t.cdf(t, df)
    else:  # "less"
        p = stats.t.cdf(t, df)

    return t, df, p, se

def fishers_exact(x1, n1, x2, n2, alternative="two-sided"):
    """
    Fisher's exact test on 2x2 table:
      [x1, n1-x1]
      [x2, n2-x2]
    SciPy alternative: 'two-sided', 'less', 'greater'
    """
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    # Map alternatives
    alt_map = {"two-sided": "two-sided", "greater": "greater", "less": "less"}
    oddsratio, p = stats.fisher_exact(table, alternative=alt_map[alternative])
    return oddsratio, p

def wald_ci_diff(x1, n1, x2, n2, alpha=0.05):
    """
    Wald CI for (p2 - p1) using unpooled SE.
    """
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = norm_ppf(1 - alpha / 2)
    lo = diff - z * se
    hi = diff + z * se
    return diff, lo, hi, se

def wilson_ci_single(x, n, alpha=0.05):
    """
    Wilson score interval for a single proportion.
    """
    if n == 0:
        return np.nan, np.nan
    z = norm_ppf(1 - alpha / 2)
    p = x / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    return max(0.0, center - half), min(1.0, center + half)

def newcombe_ci_diff(x1, n1, x2, n2, alpha=0.05):
    """
    Newcombe CI for difference in proportions using Wilson intervals:
    CI(p2 - p1) = [L2 - U1, U2 - L1]
    """
    l1, u1 = wilson_ci_single(x1, n1, alpha)
    l2, u2 = wilson_ci_single(x2, n2, alpha)
    diff = (x2 / n2) - (x1 / n1)
    return diff, (l2 - u1), (u2 - l1)

def format_pct(x):
    return f"{x:.2%}" if np.isfinite(x) else "NA"

def format_num(x, decimals=3):
    return f"{x:.{decimals}f}" if np.isfinite(x) else "NA"

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("🎛️ Inputs")
    with st.expander("Observed counts", expanded=True):
        col1, col2 = st.columns(2)
        x1 = col1.number_input("Control clicks (x1)", value=100, min_value=0, step=1)
        n1 = col1.number_input("Control views (n1)", value=1000, min_value=1, step=1)
        x2 = col2.number_input("Variant clicks (x2)", value=130, min_value=0, step=1)
        n2 = col2.number_input("Variant views (n2)", value=1100, min_value=1, step=1)

    with st.expander("Test settings", expanded=True):
        alpha = st.slider("Significance α", 0.001, 0.20, 0.05, 0.001)
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"], index=0)
        mde = st.slider("Meaningful lift threshold (Δ CTR)", 0.0, 0.05, 0.005, 0.0005)

    with st.expander("Simulation (optional)", expanded=False):
        run_sim = st.toggle("Run simulation diagnostics", value=False)
        sim_mode = st.selectbox("Simulation mode", ["Under H0 (no effect)", "Under H1 (effect)"], index=0)
        B = st.slider("Simulations (B)", 200, 5000, 1500, 100)
        base_p = st.slider("Base CTR p", 0.001, 0.30, 0.10, 0.001)
        effect = st.slider("Effect (absolute Δ)", 0.0, 0.05, 0.005, 0.0005)
        sim_seed = st.number_input("Sim seed", value=7, min_value=0, step=1)

# -------------------------
# Validate input
# -------------------------
validate_counts(int(x1), int(n1), "Control")
validate_counts(int(x2), int(n2), "Variant")

x1, n1, x2, n2 = int(x1), int(n1), int(x2), int(n2)

p1 = x1 / n1
p2 = x2 / n2
diff = p2 - p1
rel = diff / p1 if p1 > 0 else np.nan

# -------------------------
# Compute tests & intervals
# -------------------------
z_stat, z_p, z_se = two_prop_ztest(x1, n1, x2, n2, alternative=alternative)
t_stat, t_df, t_p, t_se = welch_ttest_bernoulli(x1, n1, x2, n2, alternative=alternative)
odds, f_p = fishers_exact(x1, n1, x2, n2, alternative=alternative)

wald_diff, wald_lo, wald_hi, wald_se = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
newc_diff, newc_lo, newc_hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)

sig_z = (z_p <= alpha) if np.isfinite(z_p) else False
sig_t = (t_p <= alpha) if np.isfinite(t_p) else False
sig_f = (f_p <= alpha) if np.isfinite(f_p) else False

# -------------------------
# Top KPIs
# -------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Control CTR", format_pct(p1), f"{x1}/{n1}")
with k2:
    st.metric("Variant CTR", format_pct(p2), f"{x2}/{n2}")
with k3:
    st.metric("Δ CTR (B − A)", format_pct(diff), f"Rel: {format_pct(rel) if np.isfinite(rel) else 'NA'}")
with k4:
    st.metric("α", f"{alpha:.3f}", f"Alt: {alternative}")

st.divider()

# -------------------------
# Results table
# -------------------------
st.subheader("Inference Results (p-values + Confidence Intervals)")

results = pd.DataFrame([
    {
        "Method": "Two-proportion z-test (pooled)",
        "Test stat": format_num(z_stat),
        "df": "—",
        "p-value": format_num(z_p, 6),
        f"Sig (p≤α)": "✅" if sig_z else "—",
        f"P(lift > MDE)?": "—",
        "Notes": "Normal approximation; good at large n"
    },
    {
        "Method": "Welch t-test on 0/1 outcomes",
        "Test stat": format_num(t_stat),
        "df": format_num(t_df, 1),
        "p-value": format_num(t_p, 6),
        f"Sig (p≤α)": "✅" if sig_t else "—",
        f"P(lift > MDE)?": "—",
        "Notes": "Often close to z when n large; can differ at small n"
    },
    {
        "Method": "Fisher's exact test",
        "Test stat": f"OR={format_num(odds, 3)}",
        "df": "—",
        "p-value": format_num(f_p, 6),
        f"Sig (p≤α)": "✅" if sig_f else "—",
        f"P(lift > MDE)?": "—",
        "Notes": "Exact for small counts; conservative sometimes"
    }
])

st.dataframe(results, use_container_width=True, hide_index=True)

ci_tbl = pd.DataFrame([
    {
        "CI method": f"Wald CI for Δ (α={alpha:.3f})",
        "Δ estimate": format_pct(wald_diff),
        "Lower": format_pct(wald_lo),
        "Upper": format_pct(wald_hi),
        "Note": "Simple; can be inaccurate at small n or extreme CTR"
    },
    {
        "CI method": f"Newcombe (Wilson) CI for Δ (α={alpha:.3f})",
        "Δ estimate": format_pct(newc_diff),
        "Lower": format_pct(newc_lo),
        "Upper": format_pct(newc_hi),
        "Note": "Usually more reliable than Wald"
    }
])
st.dataframe(ci_tbl, use_container_width=True, hide_index=True)

st.divider()

# -------------------------
# Visual comparisons
# -------------------------
tab1, tab2 = st.tabs(["📉 CIs side-by-side", "🧪 Simulation diagnostics (optional)"])

with tab1:
    st.subheader("Confidence intervals for Δ CTR (B − A)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[wald_lo, wald_hi],
        y=["Wald", "Wald"],
        mode="lines",
        name="Wald CI"
    ))
    fig.add_trace(go.Scatter(
        x=[newc_lo, newc_hi],
        y=["Newcombe/Wilson", "Newcombe/Wilson"],
        mode="lines",
        name="Newcombe/Wilson CI"
    ))
    fig.add_trace(go.Scatter(
        x=[diff],
        y=["Observed Δ"],
        mode="markers",
        name="Observed Δ",
        marker=dict(size=10)
    ))
    fig.update_layout(
        xaxis_title="Δ CTR",
        yaxis_title="",
        height=320,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "If you reduce n or push CTR toward extremes, Wald often misbehaves; Newcombe/Wilson stays more stable."
    )

with tab2:
    st.subheader("Simulation diagnostics")
    if not run_sim:
        st.info("Toggle **Run simulation diagnostics** in the sidebar to compare methods across many random experiments.")
    else:
        rng = np.random.default_rng(int(sim_seed))

        if sim_mode == "Under H0 (no effect)":
            pA = base_p
            pB = base_p
        else:
            pA = base_p
            pB = min(0.999, max(0.001, base_p + effect))

        z_ps, t_ps, f_ps = [], [], []
        wald_cover, newc_cover = 0, 0

        true_diff = pB - pA

        for _ in range(int(B)):
            xa = rng.binomial(n1, pA)
            xb = rng.binomial(n2, pB)

            z_stat_s, z_p_s, _ = two_prop_ztest(xa, n1, xb, n2, alternative="two-sided")
            t_stat_s, t_df_s, t_p_s, _ = welch_ttest_bernoulli(xa, n1, xb, n2, alternative="two-sided")
            _, f_p_s = fishers_exact(xa, n1, xb, n2, alternative="two-sided")

            z_ps.append(z_p_s if np.isfinite(z_p_s) else 1.0)
            t_ps.append(t_p_s if np.isfinite(t_p_s) else 1.0)
            f_ps.append(f_p_s if np.isfinite(f_p_s) else 1.0)

            # CI coverage (two-sided)
            _, lo_w, hi_w, _ = wald_ci_diff(xa, n1, xb, n2, alpha=alpha)
            _, lo_n, hi_n = newcombe_ci_diff(xa, n1, xb, n2, alpha=alpha)
            if lo_w <= true_diff <= hi_w:
                wald_cover += 1
            if lo_n <= true_diff <= hi_n:
                newc_cover += 1

        z_ps = np.array(z_ps)
        t_ps = np.array(t_ps)
        f_ps = np.array(f_ps)

        fp_z = float(np.mean(z_ps <= alpha))
        fp_t = float(np.mean(t_ps <= alpha))
        fp_f = float(np.mean(f_ps <= alpha))

        cov_w = wald_cover / B
        cov_n = newc_cover / B

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Reject rate (z-test)", f"{fp_z:.1%}")
        with c2:
            st.metric("Reject rate (t-test)", f"{fp_t:.1%}")
        with c3:
            st.metric("Reject rate (Fisher)", f"{fp_f:.1%}")
        with c4:
            st.metric("True Δ", f"{true_diff:+.2%}")

        st.caption(
            "Under **H0**, reject rate ≈ Type I error (should be near α). Under **H1**, reject rate ≈ power."
        )

        # p-value histograms (overlay-ish via bars)
        figp = go.Figure()
        bins = np.linspace(0, 1, 21)

        figp.add_trace(go.Histogram(x=z_ps, name="z-test p", nbinsx=20, opacity=0.6))
        figp.add_trace(go.Histogram(x=t_ps, name="t-test p", nbinsx=20, opacity=0.6))
        figp.add_trace(go.Histogram(x=f_ps, name="Fisher p", nbinsx=20, opacity=0.6))

        figp.update_layout(
            barmode="overlay",
            xaxis_title="p-value",
            yaxis_title="count",
            height=360
        )
        st.plotly_chart(figp, use_container_width=True)

        ci_cov_df = pd.DataFrame([
            {"CI method": "Wald", "Empirical coverage": f"{cov_w:.1%}", "Target": f"{1-alpha:.1%}"},
            {"CI method": "Newcombe/Wilson", "Empirical coverage": f"{cov_n:.1%}", "Target": f"{1-alpha:.1%}"}
        ])
        st.dataframe(ci_cov_df, use_container_width=True, hide_index=True)

        st.info(
            "Tip: try small n (e.g., 200) and extreme CTR (e.g., 0.5% or 20%) to see methods diverge more."
        )
