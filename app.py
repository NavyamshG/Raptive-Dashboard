import streamlit as st
import numpy as np
import plotly.graph_objects as 
import scipy.stats as stats
import math

# ============================================================
# CTR Inference Lab: Comparative Statistical Analysis
# ============================================================

st.set_page_config(page_title="CTR Inference Lab", layout="wide")

# Custom CSS for high-impact headers and professional text blocks
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }

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
    """,
    unsafe_allow_html=True,
)

# Main Title with Gradient
st.write(
    """
    <h1 style="font-size: 5rem; font-weight: 900; margin-bottom: 0px;">
        <span style="
            background: linear-gradient(to right, #1E1E1E, #ff4b4b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        "> CTR Inference Lab</span>
    </h1>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="sub-header">Evaluating Estimator Robustness: How Distributional Assumptions Impact Significance in Discrete A/B Testing</p>',
    unsafe_allow_html=True,
)

# Dashboard Summary & Explanation
st.markdown(
    """
<div class="description-box">
    <strong>Objective:</strong>
    Evaluating p-value and confidence interval variance across Bernoulli and Binomial modeling frameworks to compare the sensitivity of Z-test, Welch’s T-test, and Fisher’s Exact methods.
    <hr style="margin: 15px 0; border: 0; border-top: 1px solid #ddd;">
    <div class="explanation-text">
        <strong>Description:</strong><br>
        This lab demonstrates how different mathematical "lenses" interpret the same data, comparing the standard bell-curve approximations of Z and T-tests against the exact probability calculations of Fisher’s method. It reveals whether a statistical "win" is a robust result or simply a byproduct of the specific distribution and test selected.
    </div>
</div>
""",
    unsafe_allow_html=True,
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
    p1_, p2_ = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p2_ - p1_) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def welch_ttest_bernoulli(x1, n1, x2, n2):
    p1_, p2_ = x1 / n1, x2 / n2
    if n1 <= 1 or n2 <= 1:
        return np.nan, np.nan, np.nan
    s1_sq = (n1 / (n1 - 1)) * p1_ * (1 - p1_)
    s2_sq = (n2 / (n2 - 1)) * p2_ * (1 - p2_)
    se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    if se == 0:
        return np.nan, np.nan, np.nan
    t = (p2_ - p1_) / se
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
    p1_, p2_ = x1 / n1, x2 / n2
    d = p2_ - p1_
    se = math.sqrt(p1_ * (1 - p1_) / n1 + p2_ * (1 - p2_) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return d, d - z * se, d + z * se


def wilson_ci_single(x, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
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


def beta_credible_interval(samples, level=0.95):
    lo = (1 - level) / 2
    hi = 1 - lo
    return np.percentile(samples, [100 * lo, 100 * hi])


def hdi(samples, level=0.95):
    x = np.sort(np.asarray(samples))
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    interval_idx = int(np.floor(level * n))
    interval_idx = min(max(interval_idx, 1), n - 1)
    widths = x[interval_idx:] - x[: n - interval_idx]
    min_idx = np.argmin(widths)
    return x[min_idx], x[min_idx + interval_idx]


def normal_pdf(x, mu, sigma):
    if sigma <= 0 or not np.isfinite(sigma):
        return np.zeros_like(x)
    return stats.norm.pdf(x, loc=mu, scale=sigma)


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("🕹️ Scenario Selector")
    scenario = st.selectbox(
        "Choose a scenario:", ["Manual Entry", "Small Sample", "Marginal Win", "Clear Winner"]
    )
    presets = {
        "Manual Entry": (20, 200, 35, 200),
        "Small Sample": (2, 20, 5, 20),
        "Marginal Win": (100, 1000, 125, 1000),
        "Clear Winner": (50, 1000, 120, 1000),
    }
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

    # -------------------------------------------------
    # Bayesian Settings
    # -------------------------------------------------
    st.divider()
    st.header("🧠 Bayesian Settings")

    with st.expander("Prior Setup"):
        prior_mode = st.selectbox(
            "Choose prior type:",
            ["Uniform (Beta(1,1))", "Jeffreys (Beta(0.5,0.5))", "Custom Beta(a,b)"],
        )
        if prior_mode == "Custom Beta(a,b)":
            prior_a = st.number_input("Prior α (a)", min_value=0.01, value=1.0, step=0.5)
            prior_b = st.number_input("Prior β (b)", min_value=0.01, value=1.0, step=0.5)
        elif prior_mode == "Jeffreys (Beta(0.5,0.5))":
            prior_a, prior_b = 0.5, 0.5
        else:
            prior_a, prior_b = 1.0, 1.0

        st.caption("Tip: stronger priors have larger a+b.")

    with st.expander("Posterior Sampling & Interval"):
        bayes_samples = st.slider("Posterior samples", 5000, 100000, 20000, step=5000)
        credible_level = st.slider("Credible interval level", 0.80, 0.99, 0.95, step=0.01)
        show_prior_overlay = st.toggle("Show prior on effect chart", value=True)

    with st.expander("Likelihood Strength Demo"):
        st.caption(
            "This scales the effective data size while keeping the observed CTR roughly the same."
        )
        like_weight = st.slider(
            "Likelihood weight (0.1 = weak data, 1.0 = actual data, 3.0 = strong data)",
            0.1,
            3.0,
            1.0,
            step=0.1,
        )

# -------------------------
# Calculations
# -------------------------
validate_counts(int(x1), int(n1), "Control")
validate_counts(int(x2), int(n2), "Variant")

p1, p2 = x1 / n1, x2 / n2
diff = p2 - p1

z_stat, z_p = two_prop_ztest(x1, n1, x2, n2)
t_stat, t_df, t_p = welch_ttest_bernoulli(x1, n1, x2, n2)
odds, f_p = fishers_exact(x1, n1, x2, n2)

tests = {"z-test": z_p, "t-test": t_p, "Fisher": f_p}
wins = [name for name, p in tests.items() if np.isfinite(p) and p <= alpha]
losses = [name for name, p in tests.items() if np.isfinite(p) and p > alpha]

wald_d, wald_lo, wald_hi = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
newc_d, newc_lo, newc_hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)

# -------------------------------------------------
# Tabs for Frequentist vs Bayesian Analysis
# -------------------------------------------------
tab1, tab2 = st.tabs(["Frequentist Inference", "Bayesian Inference"])

# =========================================================
# TAB 1: Frequentist (DO NOT CHANGE)
# =========================================================
with tab1:
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
        y1 = stats.norm.pdf(x_axis, p1, math.sqrt(p1 * (1 - p1) / n1))
        y2 = stats.norm.pdf(x_axis, p2, math.sqrt(p2 * (1 - p2) / n2))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y1, fill="tozeroy", name="Control", line_color="#636EFA"))
        fig.add_trace(go.Scatter(x=x_axis, y=y2, fill="tozeroy", name="Variant", line_color="#00CC96"))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_results:
        if wins:
            st.success(f"**Winner according to:** {', '.join(wins)}")
        if losses:
            st.error(f"**No Significance according to:** {', '.join(losses)}")

        st.markdown("**p-value Comparison**")
        fig_p = go.Figure(
            go.Bar(
                x=list(tests.keys()),
                y=list(tests.values()),
                marker_color=["#00CC96" if p <= alpha else "#EF553B" for p in tests.values()],
                text=[f"{p:.4f}" for p in tests.values()],
                textposition="auto",
            )
        )
        fig_p.add_hline(y=alpha, line_dash="dash")
        fig_p.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown(
        """
<div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; border: 1px solid #add8e6; margin-top: 20px;">
    <span style="font-size: 1.1rem;">✨ <strong>To see a magic:</strong>
    Set Control clicks: <strong>20</strong>, Control views: <strong>200</strong>,
    Variant clicks: <strong>35</strong>, Variant views: <strong>200</strong>,
    and Significance level: <strong>0.04</strong></span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # 2) Confidence Intervals
    st.subheader("2) Delta Confidence Intervals")

    st.markdown(
        f"""<span style="font-size: 1.2rem;">
        <strong>Wald Interval (Binomial):</strong> <code>{pct(wald_lo)}</code> to <code>{pct(wald_hi)}</code>
        </span>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<span style="font-size: 1.2rem;">
        <strong>Newcombe Interval (Wilson):</strong> <code>{pct(newc_lo)}</code> to <code>{pct(newc_hi)}</code>
        </span>""",
        unsafe_allow_html=True,
    )

    fig_ci = go.Figure()
    fig_ci.add_trace(
        go.Scatter(
            x=[diff],
            y=["Wald (Binomial)"],
            mode="markers",
            error_x=dict(
                type="data",
                array=[wald_hi - diff],
                arrayminus=[diff - wald_lo],
                visible=True,
            ),
            marker=dict(size=12, color="#AB63FA"),
        )
    )
    fig_ci.add_trace(
        go.Scatter(
            x=[diff],
            y=["Newcombe (Wilson)"],
            mode="markers",
            error_x=dict(
                type="data",
                array=[newc_hi - diff],
                arrayminus=[diff - newc_lo],
                visible=True,
            ),
            marker=dict(size=12, color="#EF553B"),
        )
    )
    fig_ci.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_ci.update_layout(height=300, xaxis_title="Abs Difference", showlegend=False)
    st.plotly_chart(fig_ci, use_container_width=True)

    st.divider()

    # 3) Peeking Danger Demo
    if show_peeking:
        st.subheader("3) The Danger of Peeking (False Positive Risk)")
        rng = np.random.default_rng(int(seed))
        n_total = int(n1 + n2)
        sample_points = np.linspace(10, n_total, looks).astype(int)

        example_p_journey = []
        a_hits = rng.binomial(1, p1, size=n_total)
        b_hits = rng.binomial(1, p1, size=n_total)
        for n_pt in sample_points:
            xa, xb = a_hits[:n_pt].sum(), b_hits[:n_pt].sum()
            _, p_val = two_prop_ztest(xa, n_pt, xb, n_pt)
            example_p_journey.append(p_val)

        any_fp = 0
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
            fig_journey = go.Figure(
                go.Scatter(
                    x=sample_points,
                    y=example_p_journey,
                    mode="lines+markers",
                    line_color="#FF4B4B",
                )
            )
            fig_journey.add_hline(y=alpha, line_dash="dash")
            fig_journey.update_layout(height=350, yaxis_title="P-Value Over Time", yaxis_range=[0, 1])
            st.plotly_chart(fig_journey, use_container_width=True)

        with col_peek_stats:
            st.metric("Actual False Positive Rate", f"{fp_rate:.1%}")
            st.info(f"Checking results {looks} times instead of once increases your error rate by {fp_rate/alpha:.1f}x.")

# =========================================================
# TAB 2: Bayesian CTR (CHANGED)
# =========================================================
with tab2:
    st.subheader("Bayesian CTR Inference (Beta-Binomial)")

    # Prior
    a0, b0 = float(prior_a), float(prior_b)

    # Likelihood-weighted counts
    x1w = int(round(float(x1) * like_weight))
    n1w = max(1, int(round(float(n1) * like_weight)))
    x2w = int(round(float(x2) * like_weight))
    n2w = max(1, int(round(float(n2) * like_weight)))

    x1w = min(max(0, x1w), n1w)
    x2w = min(max(0, x2w), n2w)

    # Posterior params
    a1_post, b1_post = a0 + x1w, b0 + (n1w - x1w)
    a2_post, b2_post = a0 + x2w, b0 + (n2w - x2w)

    # Sampling
    rng_b = np.random.default_rng(int(seed) + 7)
    control_samples = rng_b.beta(a1_post, b1_post, size=int(bayes_samples))
    variant_samples = rng_b.beta(a2_post, b2_post, size=int(bayes_samples))
    lift_samples = variant_samples - control_samples

    # Prior effect samples
    prior_control_samples = rng_b.beta(a0, b0, size=int(bayes_samples))
    prior_variant_samples = rng_b.beta(a0, b0, size=int(bayes_samples))
    prior_lift_samples = prior_variant_samples - prior_control_samples

    # Likelihood approximation on effect scale
    p1w = x1w / n1w
    p2w = x2w / n2w
    like_mu = p2w - p1w
    like_se = math.sqrt(
        max(1e-12, p1w * (1 - p1w) / n1w + p2w * (1 - p2w) / n2w)
    )

    # Posterior effect summaries
    post_mean = float(np.mean(lift_samples))
    post_lo, post_hi = beta_credible_interval(lift_samples, level=float(credible_level))
    hdi_lo, hdi_hi = hdi(lift_samples, level=float(credible_level))

    ctrl_lo, ctrl_hi = beta_credible_interval(control_samples, level=float(credible_level))
    var_lo, var_hi = beta_credible_interval(variant_samples, level=float(credible_level))
    prob_variant_wins = float(np.mean(lift_samples > 0))

    # Bayes-style support summary via Savage-Dickey style density ratio approximation
    prior_sd = float(np.std(prior_lift_samples, ddof=1))
    post_sd = float(np.std(lift_samples, ddof=1))
    prior_density_0 = stats.norm.pdf(0, loc=float(np.mean(prior_lift_samples)), scale=max(prior_sd, 1e-6))
    post_density_0 = stats.gaussian_kde(lift_samples)([0])[0] if np.std(lift_samples) > 0 else 1e6
    bf01 = float(post_density_0 / max(prior_density_0, 1e-12))
    bf10 = float(1 / max(bf01, 1e-12))

    # Top metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P(Variant > Control)", f"{prob_variant_wins:.2%}")
    k2.metric(f"Control {int(credible_level*100)}% CrI", f"{ctrl_lo:.2%} — {ctrl_hi:.2%}")
    k3.metric(f"Variant {int(credible_level*100)}% CrI", f"{var_lo:.2%} — {var_hi:.2%}")
    k4.metric(f"Lift {int(credible_level*100)}% CrI", f"{post_lo:.2%} — {post_hi:.2%}")

    st.divider()

    # 1) Effect chart like the reference image
    st.subheader("1) Posterior, Prior, and Likelihood on Effect Size")

    chart_col, metric_col = st.columns([4.8, 1.2])

    with chart_col:
        x_min = min(
            np.percentile(prior_lift_samples, 0.5),
            np.percentile(lift_samples, 0.5),
            like_mu - 4 * like_se,
            -0.05,
        )
        x_max = max(
            np.percentile(prior_lift_samples, 99.5),
            np.percentile(lift_samples, 99.5),
            like_mu + 4 * like_se,
            0.05,
        )
        xs = np.linspace(x_min, x_max, 800)

        prior_mean = float(np.mean(prior_lift_samples))
        prior_sd = max(float(np.std(prior_lift_samples, ddof=1)), 1e-6)
        post_sd = max(float(np.std(lift_samples, ddof=1)), 1e-6)

        prior_pdf = normal_pdf(xs, prior_mean, prior_sd)
        like_pdf = normal_pdf(xs, like_mu, like_se)
        post_pdf = normal_pdf(xs, post_mean, post_sd)

        y_max = max(prior_pdf.max(), like_pdf.max(), post_pdf.max())

        fig_effect = go.Figure()

        if show_prior_overlay:
            fig_effect.add_trace(
                go.Scatter(
                    x=xs,
                    y=prior_pdf,
                    mode="lines",
                    name="prior",
                    line=dict(color="black", width=3, dash="dot"),
                )
            )

        fig_effect.add_trace(
            go.Scatter(
                x=xs,
                y=post_pdf,
                mode="lines",
                name="posterior",
                line=dict(color="#C23B22", width=3),
            )
        )

        fig_effect.add_trace(
            go.Scatter(
                x=xs,
                y=like_pdf,
                mode="lines",
                name="likelihood",
                line=dict(color="#3B83CC", width=3, dash="dash"),
            )
        )

        # Mark means
        if show_prior_overlay:
            fig_effect.add_trace(
                go.Scatter(
                    x=[prior_mean],
                    y=[normal_pdf(np.array([prior_mean]), prior_mean, prior_sd)[0]],
                    mode="markers",
                    marker=dict(color="white", size=10, line=dict(color="black", width=2)),
                    showlegend=False,
                )
            )

        fig_effect.add_trace(
            go.Scatter(
                x=[post_mean],
                y=[normal_pdf(np.array([post_mean]), post_mean, post_sd)[0]],
                mode="markers",
                marker=dict(color="white", size=10, line=dict(color="#C23B22", width=2)),
                showlegend=False,
            )
        )

        fig_effect.add_trace(
            go.Scatter(
                x=[like_mu],
                y=[normal_pdf(np.array([like_mu]), like_mu, like_se)[0]],
                mode="markers",
                marker=dict(color="white", size=10, line=dict(color="#3B83CC", width=2)),
                showlegend=False,
            )
        )

        # Labels over curves
        if show_prior_overlay:
            fig_effect.add_annotation(x=prior_mean, y=y_max * 1.08, text="prior", showarrow=False, font=dict(size=14, color="black"))
        fig_effect.add_annotation(x=post_mean, y=y_max * 1.12, text="posterior", showarrow=False, font=dict(size=14, color="#C23B22"))
        fig_effect.add_annotation(x=like_mu, y=y_max * 1.08, text="likelihood", showarrow=False, font=dict(size=14, color="#3B83CC"))

        # Zero reference
        fig_effect.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        # Frequentist CI and Bayesian HDI lines near bottom
        y_ci = y_max * -0.10
        y_hdi = y_max * -0.18

        fig_effect.add_shape(type="line", x0=wald_lo, x1=wald_hi, y0=y_ci, y1=y_ci, line=dict(color="#3B83CC", width=3))
        fig_effect.add_shape(type="line", x0=hdi_lo, x1=hdi_hi, y0=y_hdi, y1=y_hdi, line=dict(color="#C23B22", width=3))

        fig_effect.add_annotation(
            x=(wald_lo + wald_hi) / 2,
            y=y_ci + y_max * 0.02,
            text=f"{int(credible_level*100)} % CI [{wald_lo:.2%}, {wald_hi:.2%}]",
            showarrow=False,
            font=dict(size=12, color="black"),
        )
        fig_effect.add_annotation(
            x=(hdi_lo + hdi_hi) / 2,
            y=y_hdi + y_max * 0.02,
            text=f"{int(credible_level*100)} % HDI [{hdi_lo:.2%}, {hdi_hi:.2%}]",
            showarrow=False,
            font=dict(size=12, color="black"),
        )

        fig_effect.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Effect (Variant CTR − Control CTR)",
            yaxis_title="Density",
            yaxis=dict(range=[y_max * -0.22, y_max * 1.18]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_effect, use_container_width=True)

    with metric_col:
        st.metric("Support H₀ (BF₀₁)", f"{bf01:.3f}")
        st.metric("Support H₁ (BF₁₀)", f"{bf10:.3f}")
        st.metric("p-value", f"{z_p:.4f}")
        st.caption(
            "BF values here are an effect-scale approximation, useful for intuition in the dashboard."
        )

    st.markdown(
        f"""
- **Prior:** Beta({a0:g}, {b0:g})
- **Likelihood weight:** {like_weight:.1f}×
- **Posterior (Control):** Beta({a1_post:.1f}, {b1_post:.1f})
- **Posterior (Variant):** Beta({a2_post:.1f}, {b2_post:.1f})
""".strip()
    )

    st.divider()

    # 2) Bayesian credible intervals
    st.subheader("2) Bayesian Credible Intervals")

    st.markdown(
        f"""<span style="font-size: 1.2rem;">
        <strong>Lift Credible Interval ({int(credible_level*100)}%):</strong>
        <code>{post_lo:.2%}</code> to <code>{post_hi:.2%}</code>
        </span>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<span style="font-size: 1.2rem;">
        <strong>Lift HDI ({int(credible_level*100)}%):</strong>
        <code>{hdi_lo:.2%}</code> to <code>{hdi_hi:.2%}</code>
        </span>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<span style="font-size: 1.2rem;">
        <strong>Control CTR Credible Interval ({int(credible_level*100)}%):</strong>
        <code>{ctrl_lo:.2%}</code> to <code>{ctrl_hi:.2%}</code>
        </span>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<span style="font-size: 1.2rem;">
        <strong>Variant CTR Credible Interval ({int(credible_level*100)}%):</strong>
        <code>{var_lo:.2%}</code> to <code>{var_hi:.2%}</code>
        </span>""",
        unsafe_allow_html=True,
    )

    lift_mean = post_mean
    fig_bci = go.Figure()
    fig_bci.add_trace(
        go.Scatter(
            x=[lift_mean],
            y=[f"Lift ({int(credible_level*100)}% CrI)"],
            mode="markers",
            error_x=dict(
                type="data",
                array=[post_hi - lift_mean],
                arrayminus=[lift_mean - post_lo],
                visible=True,
            ),
            marker=dict(size=12, color="#FF4B4B"),
        )
    )
    fig_bci.add_trace(
        go.Scatter(
            x=[lift_mean],
            y=[f"Lift ({int(credible_level*100)}% HDI)"],
            mode="markers",
            error_x=dict(
                type="data",
                array=[hdi_hi - lift_mean],
                arrayminus=[lift_mean - hdi_lo],
                visible=True,
            ),
            marker=dict(size=12, color="#C23B22"),
        )
    )
    fig_bci.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_bci.update_layout(
        height=280,
        xaxis_title="Abs Lift",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_bci, use_container_width=True)

    st.divider()

    # 3) Lift posterior
    st.subheader("3) Lift (Variant − Control) Posterior")

    fig_lift = go.Figure()
    fig_lift.add_trace(go.Histogram(x=lift_samples, nbinsx=70))
    fig_lift.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_lift.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Absolute Lift",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    st.info(
        "Interpretation: the effect chart separates prior belief, data likelihood, and posterior belief on the same effect scale. "
        "As you strengthen the prior or change likelihood weight, the posterior shifts accordingly."
    )
