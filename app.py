import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab (Simple + Intuitive)
# 1) Same data, different tests (z vs Fisher) -> different p-values
# 2) Peeking inflates false positives under H0
# ============================================================

st.set_page_config(page_title="CTR Inference Lab (Simple)", layout="wide")
st.title("📊 CTR Inference Lab (Simple)")
st.caption("Two takeaways: (1) different tests can disagree at small n, (2) peeking inflates false positives.")

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

def two_prop_z_pvalue(x1, n1, x2, n2):
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan
    z = ((x2 / n2) - (x1 / n1)) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))

def fishers_pvalue(x1, n1, x2, n2):
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    _, p = stats.fisher_exact(table, alternative="two-sided")
    return p

def pct(x):
    return f"{x:.2%}"

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("🎛️ Inputs")

    st.subheader("A/B counts")
    col1, col2 = st.columns(2)
    x1 = col1.number_input("Control clicks", value=20, min_value=0, step=1)
    n1 = col1.number_input("Control views", value=200, min_value=1, step=1)
    x2 = col2.number_input("Variant clicks", value=30, min_value=0, step=1)
    n2 = col2.number_input("Variant views", value=200, min_value=1, step=1)

    st.divider()

    st.subheader("Peeking demo")
    show_peeking = st.toggle("Show peeking demo", value=True)
    alpha = st.slider("Significance α", 0.01, 0.20, 0.05, 0.01)
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

z_p = two_prop_z_pvalue(x1, n1, x2, n2)
f_p = fishers_pvalue(x1, n1, x2, n2)

# -------------------------
# Top KPIs
# -------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Control CTR", pct(p1), f"{x1}/{n1}")
with c2:
    st.metric("Variant CTR", pct(p2), f"{x2}/{n2}")
with c3:
    st.metric("Δ CTR", pct(diff))
with c4:
    st.metric("α", f"{alpha:.2f}")

st.divider()

# -------------------------
# Part 1: Same data, different tests
# -------------------------
st.subheader("1) Same data, different tests")

left, right = st.columns([1.2, 1])

with left:
    st.markdown(
        """
**Why this matters:**  
For CTR, we often use approximations. At small sample sizes or low CTRs, methods can disagree.

We show:
- **z-test** (normal approximation)
- **Fisher exact** (exact for 2×2 table)

As n grows, they usually get closer.
        """
    )

with right:
    st.markdown("**p-values**")
    st.metric("z-test p-value", "NA" if np.isnan(z_p) else f"{z_p:.4f}", "✅ sig" if (np.isfinite(z_p) and z_p <= alpha) else "—")
    st.metric("Fisher p-value", f"{f_p:.4f}", "✅ sig" if f_p <= alpha else "—")

# Mini visualization: p-values vs method
fig_methods = go.Figure()
fig_methods.add_trace(go.Bar(
    x=["z-test", "Fisher"],
    y=[z_p if np.isfinite(z_p) else 1.0, f_p],
    text=[f"{z_p:.4f}" if np.isfinite(z_p) else "NA", f"{f_p:.4f}"],
    textposition="auto"
))
fig_methods.add_hline(y=alpha)
fig_methods.update_layout(
    yaxis_title="p-value",
    height=320,
    showlegend=False
)
st.plotly_chart(fig_methods, use_container_width=True)
st.caption("The horizontal line is α. Bars below α are “statistically significant.”")

st.divider()

# -------------------------
# Part 2: Peeking demo (very simple + intuitive)
# -------------------------
if show_peeking:
    st.subheader("2) Peeking inflates false positives")

    st.markdown(
        """
**Setup:** There is **no real difference** (A and B have the same true CTR).  
But if you check results many times and stop the first time p ≤ α, you will “find” wins more often than α.
        """
    )

    rng = np.random.default_rng(int(seed))
    look_ns = np.unique(np.round(np.linspace(final_n / looks, final_n, looks)).astype(int))

    # Run experiments under H0, record if we ever cross alpha
    any_fp = 0
    first_hit = []

    for _ in range(int(sims)):
        a = rng.binomial(1, base_p, size=final_n)
        b = rng.binomial(1, base_p, size=final_n)

        hit_at = None
        for i, nlook in enumerate(look_ns, start=1):
            xa = int(a[:nlook].sum())
            xb = int(b[:nlook].sum())
            p = two_prop_z_pvalue(xa, nlook, xb, nlook)
            if np.isfinite(p) and p <= alpha:
                hit_at = i
                break

        if hit_at is not None:
            any_fp += 1
            first_hit.append(hit_at)

    fp_rate = any_fp / sims

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("False positive rate (with peeking)", f"{fp_rate:.1%}")
    with colB:
        st.metric("Target (no peeking)", f"{alpha:.1%}")
    with colC:
        approx = 1 - (1 - alpha) ** looks
        st.metric("Rough intuition", f"{approx:.1%}")

    fig_peek = go.Figure()
    if len(first_hit) > 0:
        fig_peek.add_trace(go.Histogram(
            x=first_hit,
            nbinsx=min(looks, 20)
        ))
    fig_peek.update_layout(
        xaxis_title="Which look triggered significance first (1..k)",
        yaxis_title="count",
        height=320
    )
    st.plotly_chart(fig_peek, use_container_width=True)

    st.caption(
        "If the false positive rate is well above α, that’s the peeking problem. "
        "Fixes exist (group sequential tests, alpha spending, always-valid methods), but this demo shows the issue."
    )
