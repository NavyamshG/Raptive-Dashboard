import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab
# - z-test vs t-test vs Fisher (Bernoulli/Binomial CTR)
# - Wald CI vs Newcombe/Wilson CI
# - Method disagreement vs sample size
# - OPTIONAL: peeking (sequential looks) inflates false positives
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")
st.title("📊 CTR Inference Lab")
st.caption("CTR inference on Bernoulli/Binomial data: z-test vs t-test vs Fisher, CI behavior, method disagreement, and peeking risk.")

with st.expander("📘 What this demonstrates", expanded=True):
    st.markdown(
        """
**Data model:** CTR is a Bernoulli rate (click/no-click), aggregated as Binomial counts.

**Why methods differ:**
- **Two-proportion z-test** uses a normal approximation (best at large n).
- **Welch t-test on 0/1 data** treats Bernoulli as continuous; often similar at large n, can differ at small n.
- **Fisher’s exact test** is exact for small counts.

**Confidence intervals:**
- **Wald CI** (naive) can be inaccurate for small n or extreme CTRs.
- **Newcombe CI** (Wilson score based) is usually more reliable.

**Peeking risk (sequential looks):**
If you test every day and stop when p ≤ α, your overall false positive rate becomes **much higher than α**.
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
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan, se
    z = (p2 - p1) / se

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - stats.norm.cdf(z)
    else:
        p = stats.norm.cdf(z)
    return z, p, se

def welch_ttest_bernoulli(x1, n1, x2, n2, alternative="two-sided"):
    p1 = x1 / n1
    p2 = x2 / n2
    if n1 <= 1 or n2 <= 1:
        return np.nan, np.nan, np.nan, np.nan

    s1_sq = (n1 / (n1 - 1)) * p1 * (1 - p1)
    s2_sq = (n2 / (n2 - 1)) * p2 * (1 - p2)

    se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    if se == 0:
        return np.nan, np.nan, np.nan, se

    t = (p2 - p1) / se
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else np.nan

    if alternative == "two-sided":
        p = 2 * (1 - stats.t.cdf(abs(t), df))
    elif alternative == "greater":
        p = 1 - stats.t.cdf(t, df)
    else:
        p = stats.t.cdf(t, df)

    return t, df, p, se

def fishers_exact(x1, n1, x2, n2, alternative="two-sided"):
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    alt_map = {"two-sided": "two-sided", "greater": "greater", "less": "less"}
    oddsratio, p = stats.fisher_exact(table, alternative=alt_map[alternative])
    return oddsratio, p

def wald_ci_diff(x1, n1, x2, n2, alpha=0.05):
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = norm_ppf(1 - alpha / 2)
    lo = diff - z * se
    hi = diff + z * se
    return diff, lo, hi, se

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

    with st.expander("Simulation diagnostics (optional)", expanded=False):
        run_sim = st.toggle("Run simulation diagnostics", value=False)
        sim_mode = st.selectbox("Simulation mode", ["Under H0 (no effect)", "Under H1 (effect)"], index=0)
        B = st.slider("Simulations (B)", 200, 5000, 1500, 100)
        base_p = st.slider("Base CTR p", 0.001, 0.30, 0.10, 0.001)
        effect = st.slider("Effect (absolute Δ)", 0.0, 0.05, 0.005, 0.0005)
        sim_seed = st.number_input("Sim seed", value=7, min_value=0, step=1)

    with st.expander("Impactful viz settings", expanded=False):
        show_disagreement = st.toggle("Show method disagreement vs sample size", value=True)
        max_n = st.slider("Max sample size per arm", 500, 50000, 20000, 500)
        n_points = st.slider("Points on curve", 10, 60, 30, 1)
        curve_seed = st.number_input("Curve seed", value=123, min_value=0, step=1)

    with st.expander("🚨 Peeking risk (sequential looks)", expanded=False):
        show_peeking = st.toggle("Show peeking risk demo", value=True)
        looks = st.slider("Number of interim looks (k)", 2, 50, 20, 1)
        final_n = st.slider("Final sample size per arm", 200, 200000, 20000, 200)
        peeking_B = st.slider("Simulations for peeking", 200, 5000, 1500, 100)
        peek_base_p = st.slider("Base CTR (H0) for peeking", 0.001, 0.30, 0.10, 0.001)
        peeking_seed = st.number_input("Peeking seed", value=99, min_value=0, step=1)

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
z_stat, z_p, _ = two_prop_ztest(x1, n1, x2, n2, alternative=alternative)
t_stat, t_df, t_p, _ = welch_ttest_bernoulli(x1, n1, x2, n2, alternative=alternative)
odds, f_p = fishers_exact(x1, n1, x2, n2, alternative=alternative)

wald_diff, wald_lo, wald_hi, _ = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
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
# One essential table: p-values
# -------------------------
st.subheader("Inference Results (p-values)")
results = pd.DataFrame([
    {
        "Method": "Two-proportion z-test (pooled)",
        "Test stat": format_num(z_stat),
        "df": "—",
        "p-value": format_num(z_p, 6),
        "Sig (p≤α)": "✅" if sig_z else "—",
        "Notes": "Normal approximation; good at large n"
    },
    {
        "Method": "Welch t-test on 0/1 outcomes",
        "Test stat": format_num(t_stat),
        "df": format_num(t_df, 1),
        "p-value": format_num(t_p, 6),
        "Sig (p≤α)": "✅" if sig_t else "—",
        "Notes": "Often close to z when n large; can differ at small n"
    },
    {
        "Method": "Fisher's exact test",
        "Test stat": f"OR={format_num(odds, 3)}",
        "df": "—",
        "p-value": format_num(f_p, 6),
        "Sig (p≤α)": "✅" if sig_f else "—",
        "Notes": "Exact for small counts; conservative sometimes"
    }
])
st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()

# -------------------------
# Tabs
# -------------------------
tabs = ["📉 CIs side-by-side", "🧪 Simulation diagnostics (optional)"]
if show_disagreement:
    tabs.append("📈 Method disagreement vs sample size")
if show_peeking:
    tabs.append("🚨 Peeking inflates false positives")

tab_objs = st.tabs(tabs)

# --- TAB 1: CI plot ---
with tab_objs[0]:
    st.subheader("Confidence intervals for Δ CTR (B − A)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[wald_lo, wald_hi], y=["Wald", "Wald"], mode="lines", name="Wald CI"))
    fig.add_trace(go.Scatter(x=[newc_lo, newc_hi], y=["Newcombe/Wilson", "Newcombe/Wilson"], mode="lines", name="Newcombe/Wilson CI"))
    fig.add_trace(go.Scatter(x=[diff], y=["Observed Δ"], mode="markers", name="Observed Δ", marker=dict(size=10)))
    fig.update_layout(xaxis_title="Δ CTR", yaxis_title="", height=320, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Wald can be unstable for small n or extreme CTR; Newcombe/Wilson is usually more reliable.")

# --- TAB 2: simulation diagnostics ---
with tab_objs[1]:
    st.subheader("Simulation diagnostics")
    if not run_sim:
        st.info("Toggle **Run simulation diagnostics** in the sidebar to compare methods across many random experiments.")
    else:
        rng = np.random.default_rng(int(sim_seed))
        if sim_mode == "Under H0 (no effect)":
            pA, pB = base_p, base_p
        else:
            pA, pB = base_p, min(0.999, max(0.001, base_p + effect))

        z_ps, t_ps, f_ps = [], [], []
        wald_cover, newc_cover = 0, 0
        true_diff = pB - pA

        for _ in range(int(B)):
            xa = rng.binomial(n1, pA)
            xb = rng.binomial(n2, pB)

            _, pz, _ = two_prop_ztest(xa, n1, xb, n2, alternative="two-sided")
            _, _, pt, _ = welch_ttest_bernoulli(xa, n1, xb, n2, alternative="two-sided")
            _, pf = fishers_exact(xa, n1, xb, n2, alternative="two-sided")

            z_ps.append(pz if np.isfinite(pz) else 1.0)
            t_ps.append(pt if np.isfinite(pt) else 1.0)
            f_ps.append(pf if np.isfinite(pf) else 1.0)

            _, lo_w, hi_w, _ = wald_ci_diff(xa, n1, xb, n2, alpha=alpha)
            _, lo_n, hi_n = newcombe_ci_diff(xa, n1, xb, n2, alpha=alpha)
            wald_cover += int(lo_w <= true_diff <= hi_w)
            newc_cover += int(lo_n <= true_diff <= hi_n)

        z_ps = np.array(z_ps)
        t_ps = np.array(t_ps)
        f_ps = np.array(f_ps)

        reject_z = float(np.mean(z_ps <= alpha))
        reject_t = float(np.mean(t_ps <= alpha))
        reject_f = float(np.mean(f_ps <= alpha))

        cov_w = wald_cover / B
        cov_n = newc_cover / B

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Reject rate (z-test)", f"{reject_z:.1%}")
        with c2: st.metric("Reject rate (t-test)", f"{reject_t:.1%}")
        with c3: st.metric("Reject rate (Fisher)", f"{reject_f:.1%}")
        with c4: st.metric("True Δ", f"{true_diff:+.2%}")

        figp = go.Figure()
        figp.add_trace(go.Histogram(x=z_ps, name="z-test p", nbinsx=20, opacity=0.6))
        figp.add_trace(go.Histogram(x=t_ps, name="t-test p", nbinsx=20, opacity=0.6))
        figp.add_trace(go.Histogram(x=f_ps, name="Fisher p", nbinsx=20, opacity=0.6))
        figp.update_layout(barmode="overlay", xaxis_title="p-value", yaxis_title="count", height=360)
        st.plotly_chart(figp, use_container_width=True)

        ci_cov_df = pd.DataFrame([
            {"CI method": "Wald", "Empirical coverage": f"{cov_w:.1%}", "Target": f"{1-alpha:.1%}"},
            {"CI method": "Newcombe/Wilson", "Empirical coverage": f"{cov_n:.1%}", "Target": f"{1-alpha:.1%}"}
        ])
        st.dataframe(ci_cov_df, use_container_width=True, hide_index=True)

# --- TAB 3: disagreement vs sample size ---
idx = 2
if show_disagreement:
    with tab_objs[idx]:
        st.subheader("Method disagreement vs sample size")
        st.write("Repeated experiments at different n show how approximations converge as n grows.")

        rngc = np.random.default_rng(int(curve_seed))
        pA_true = max(0.0005, min(0.9995, p1))
        pB_true = max(0.0005, min(0.9995, p2))

        ns = np.unique(np.round(np.geomspace(50, max_n, n_points)).astype(int))
        inner = 200

        z_med, t_med, f_med = [], [], []
        gap_zt, gap_zf, gap_tf = [], [], []

        for n in ns:
            zps, tps, fps = [], [], []
            for _ in range(inner):
                xa = rngc.binomial(n, pA_true)
                xb = rngc.binomial(n, pB_true)
                _, pz, _ = two_prop_ztest(xa, n, xb, n, alternative="two-sided")
                _, _, pt, _ = welch_ttest_bernoulli(xa, n, xb, n, alternative="two-sided")
                _, pf = fishers_exact(xa, n, xb, n, alternative="two-sided")
                zps.append(pz if np.isfinite(pz) else 1.0)
                tps.append(pt if np.isfinite(pt) else 1.0)
                fps.append(pf if np.isfinite(pf) else 1.0)

            zps, tps, fps = np.array(zps), np.array(tps), np.array(fps)
            z_med.append(float(np.median(zps)))
            t_med.append(float(np.median(tps)))
            f_med.append(float(np.median(fps)))
            gap_zt.append(float(np.median(np.abs(zps - tps))))
            gap_zf.append(float(np.median(np.abs(zps - fps))))
            gap_tf.append(float(np.median(np.abs(tps - fps))))

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=ns, y=z_med, mode="lines+markers", name="Median p (z-test)"))
        fig_p.add_trace(go.Scatter(x=ns, y=t_med, mode="lines+markers", name="Median p (t-test)"))
        fig_p.add_trace(go.Scatter(x=ns, y=f_med, mode="lines+markers", name="Median p (Fisher)"))
        fig_p.update_layout(xaxis_title="Sample size per arm", yaxis_title="Median p-value", height=360)
        st.plotly_chart(fig_p, use_container_width=True)

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=ns, y=gap_zt, mode="lines+markers", name="Median |p_z - p_t|"))
        fig_g.add_trace(go.Scatter(x=ns, y=gap_zf, mode="lines+markers", name="Median |p_z - p_f|"))
        fig_g.add_trace(go.Scatter(x=ns, y=gap_tf, mode="lines+markers", name="Median |p_t - p_f|"))
        fig_g.update_layout(xaxis_title="Sample size per arm", yaxis_title="Median |Δ p-value|", height=360)
        st.plotly_chart(fig_g, use_container_width=True)

    idx += 1

# --- TAB 4: peeking demo ---
if show_peeking:
    with tab_objs[idx]:
        st.subheader("🚨 Peeking inflates false positives (even under H0)")
        st.write(
            "We simulate **no true effect** (pA = pB). We then 'peek' at the data multiple times. "
            "If you stop early when p ≤ α, your overall chance of a false positive becomes much larger than α."
        )

        rngp = np.random.default_rng(int(peeking_seed))

        # interim sample sizes (increasing up to final_n)
        look_ns = np.unique(np.round(np.linspace(final_n / looks, final_n, looks)).astype(int))

        # For each run, generate full sequences of Bernoulli outcomes, then compute p-values at each look.
        # We track whether any look triggers significance.
        triggered_any = {"z": 0, "t": 0, "f": 0}
        first_hit_dist = {"z": [], "t": [], "f": []}

        for _ in range(int(peeking_B)):
            # Generate full outcomes up to final_n for both arms under H0
            a_outcomes = rngp.binomial(1, peek_base_p, size=final_n)
            b_outcomes = rngp.binomial(1, peek_base_p, size=final_n)

            hit_z = hit_t = hit_f = False
            hit_z_at = hit_t_at = hit_f_at = None

            for i, nlook in enumerate(look_ns, start=1):
                xa = int(a_outcomes[:nlook].sum())
                xb = int(b_outcomes[:nlook].sum())

                _, pz, _ = two_prop_ztest(xa, nlook, xb, nlook, alternative="two-sided")
                _, _, pt, _ = welch_ttest_bernoulli(xa, nlook, xb, nlook, alternative="two-sided")
                _, pf = fishers_exact(xa, nlook, xb, nlook, alternative="two-sided")

                if (not hit_z) and np.isfinite(pz) and (pz <= alpha):
                    hit_z, hit_z_at = True, i
                if (not hit_t) and np.isfinite(pt) and (pt <= alpha):
                    hit_t, hit_t_at = True, i
                if (not hit_f) and np.isfinite(pf) and (pf <= alpha):
                    hit_f, hit_f_at = True, i

            triggered_any["z"] += int(hit_z)
            triggered_any["t"] += int(hit_t)
            triggered_any["f"] += int(hit_f)

            if hit_z_at is not None:
                first_hit_dist["z"].append(hit_z_at)
            if hit_t_at is not None:
                first_hit_dist["t"].append(hit_t_at)
            if hit_f_at is not None:
                first_hit_dist["f"].append(hit_f_at)

        fp_any_z = triggered_any["z"] / peeking_B
        fp_any_t = triggered_any["t"] / peeking_B
        fp_any_f = triggered_any["f"] / peeking_B

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Any false positive (z)", f"{fp_any_z:.1%}", f"α={alpha:.3f}")
        with c2:
            st.metric("Any false positive (t)", f"{fp_any_t:.1%}", f"k={looks}")
        with c3:
            st.metric("Any false positive (Fisher)", f"{fp_any_f:.1%}", f"final n={final_n}")
        with c4:
            st.metric("H0 CTR", format_pct(peek_base_p))

        st.caption(
            "These are the chances of seeing **at least one** significant result across interim looks under pure noise. "
            "This is why sequential tests need correction (alpha spending, group sequential, always-valid p-values, etc.)."
        )

        # Plot: first hit distribution across looks (if any)
        fig_hit = go.Figure()
        for key, name in [("z", "z-test"), ("t", "t-test"), ("f", "Fisher")]:
            if len(first_hit_dist[key]) > 0:
                fig_hit.add_trace(go.Histogram(
                    x=first_hit_dist[key],
                    name=f"First hit look ({name})",
                    nbinsx=min(looks, 20),
                    opacity=0.6
                ))
        fig_hit.update_layout(
            barmode="overlay",
            xaxis_title="Which interim look triggered first significance (1..k)",
            yaxis_title="count",
            height=360
        )
        st.plotly_chart(fig_hit, use_container_width=True)

        # Reference: theoretical "independent looks" upper bound intuition (not exact because looks are correlated)
        approx_indep = 1 - (1 - alpha) ** looks
        st.info(
            f"Back-of-envelope intuition: if looks were independent, P(any FP) ≈ 1 − (1 − α)^k = **{approx_indep:.1%}**. "
            "In reality looks are correlated, but the inflation is still real."
        )
