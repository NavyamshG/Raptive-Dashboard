import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

# -------------------------
# CONFIG (reframed to match prompt)
# -------------------------
st.set_page_config(page_title="Beta-Binomial Simulation Lab", layout="wide")
st.title("📊 Beta-Binomial Simulation Lab")
st.caption("Demonstrating an interesting statistical property of the Beta distribution: conjugacy + posterior shrinkage")

# -------------------------
# EXPLANATION (make it explicit)
# -------------------------
with st.expander("📘 What is this demonstrating?", expanded=True):
    st.markdown(
        """
**Distribution:** Beta distribution (a probability distribution over rates in \[0,1\])

**Statistical property demonstrated:**
- **Bayesian conjugacy (Beta-Binomial):** If your prior is Beta and your data is Binomial, the posterior is still Beta.
- **Posterior shrinkage:** With small samples, estimates are pulled toward the prior; with large samples, data dominates.
- **Uncertainty shrinkage:** Posterior variance decreases as sample size increases.

Use the sliders to change the prior and sample sizes, then observe how the posterior moves and tightens.
        """
    )

# -------------------------
# SIDEBAR (stats-first controls)
# -------------------------
with st.sidebar:
    st.header("🎛️ Controls")

    with st.expander("Prior (Beta) settings", expanded=True):
        hist_ctr = st.slider("Prior mean (%)", 0.01, 30.0, 10.0) / 100.0
        prior_weight = st.number_input(
            "Prior strength (effective trials)",
            value=500,
            min_value=0,
            step=50,
            help="0 disables the historical prior and falls back to Jeffreys prior (0.5, 0.5)."
        )

    with st.expander("Observed data (Binomial)", expanded=True):
        col_a, col_b = st.columns(2)

        clicks_a = col_a.number_input("Group A successes", value=100, min_value=0, step=1)
        views_a = col_a.number_input("Group A trials", value=1000, min_value=0, step=1)

        clicks_b = col_b.number_input("Group B successes", value=130, min_value=0, step=1)
        views_b = col_b.number_input("Group B trials", value=1100, min_value=0, step=1)

        prior_mode = st.radio(
            "How to apply the prior?",
            options=["Shared prior (A & B)", "Baseline-only prior (A only)"],
            help="Shared prior anchors both groups equally. Baseline-only anchors A; B uses a weak stabilizer."
        )

        weak_prior_weight = st.number_input(
            "B weak prior strength (Baseline-only mode)",
            value=2,
            min_value=0,
            step=1,
            help="Small stabilizer for B when not sharing the historical prior."
        )

    with st.expander("Simulation", expanded=False):
        n_sims = st.slider("Posterior draws", 2000, 50000, 20000, 1000)
        seed = st.number_input("Seed", value=42, min_value=0, step=1)

    with st.expander("Meaningful effect (optional)", expanded=False):
        mde_percent = st.slider(
            "Meaningful lift threshold (MDE) %",
            0.0, 10.0, 1.0, 0.1,
            help="Used only to summarize P(lift > threshold)."
        ) / 100.0


# -------------------------
# VALIDATION
# -------------------------
def validate(clicks: int, views: int, label: str) -> None:
    if views < clicks:
        st.error(f"{label}: trials must be ≥ successes (got successes={clicks}, trials={views}).")
        st.stop()
    if views == 0:
        st.error(f"{label}: trials must be > 0 to estimate a rate.")
        st.stop()

validate(clicks_a, views_a, "Group A")
validate(clicks_b, views_b, "Group B")

# -------------------------
# BAYESIAN MODEL (Beta-Binomial)
# -------------------------
alpha_hist = max(prior_weight * hist_ctr, 0.0)
beta_hist = max(prior_weight * (1.0 - hist_ctr), 0.0)

def beta_params_for_group(successes, trials, group: str):
    """
    Beta-Binomial conjugacy:
      Prior: Beta(alpha0, beta0)
      Data:  Binomial(successes | trials, p)
      Posterior: Beta(alpha0 + successes, beta0 + trials - successes)
    """
    if prior_weight == 0:
        # Jeffreys prior
        a0, b0 = 0.5, 0.5
        return successes + a0, (trials - successes) + b0

    if prior_mode == "Shared prior (A & B)":
        a0, b0 = alpha_hist, beta_hist
        return successes + a0, (trials - successes) + b0

    # Baseline-only: A gets historical prior, B gets weak prior
    if group == "A":
        a0, b0 = alpha_hist, beta_hist
        return successes + a0, (trials - successes) + b0
    else:
        a0 = max(weak_prior_weight * hist_ctr, 0.0) + 0.5
        b0 = max(weak_prior_weight * (1.0 - hist_ctr), 0.0) + 0.5
        return successes + a0, (trials - successes) + b0

rng = np.random.default_rng(int(seed))

a_alpha, a_beta = beta_params_for_group(clicks_a, views_a, "A")
b_alpha, b_beta = beta_params_for_group(clicks_b, views_b, "B")

sim_a = rng.beta(a_alpha, a_beta, int(n_sims))
sim_b = rng.beta(b_alpha, b_beta, int(n_sims))

# -------------------------
# HELPERS
# -------------------------
def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))

a_ci = ci(sim_a)
b_ci = ci(sim_b)

a_mean = float(np.mean(sim_a))
b_mean = float(np.mean(sim_b))

# Interesting probability property: P(B > A)
p_b_gt_a = float(np.mean(sim_b > sim_a))

# Lift (optional summaries)
lift = (sim_b - sim_a) / np.clip(sim_a, 1e-9, None)
lift_ci = ci(lift)
lift_med = float(np.median(lift))
p_lift_pos = float(np.mean(lift > 0))
p_lift_gt_mde = float(np.mean(lift > mde_percent))

# -------------------------
# TOP SUMMARY (stats-first)
# -------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Posterior mean (A)", f"{a_mean:.2%}")
    st.caption(f"95% CI: [{a_ci[0]:.2%}, {a_ci[1]:.2%}]")

with k2:
    st.metric("Posterior mean (B)", f"{b_mean:.2%}")
    st.caption(f"95% CI: [{b_ci[0]:.2%}, {b_ci[1]:.2%}]")

with k3:
    st.metric("P(B > A)", f"{p_b_gt_a:.1%}")
    st.caption("Probability one Beta draw exceeds another")

with k4:
    st.metric("Median lift", f"{lift_med:+.2%}")
    st.caption(f"95% CI: [{lift_ci[0]:+.2%}, {lift_ci[1]:+.2%}]")

st.divider()

# -------------------------
# TABS (distribution + properties)
# -------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈 Prior vs Posterior", "📉 Uncertainty Shrinks with Sample Size", "🧪 Probability Outcomes"]
)

with tab1:
    st.subheader("Beta-Binomial Conjugacy: Prior → Posterior (still Beta)")
    st.write("The posterior remains a Beta distribution after observing Binomial data (conjugacy).")

    x_max = max(sim_a.max(), sim_b.max(), hist_ctr) * 1.5
    x_max = min(x_max, 1.0)
    x_range = np.linspace(0, max(0.02, x_max), 600)

    fig = go.Figure()

    if prior_weight > 0:
        prior_pdf = stats.beta.pdf(x_range, max(alpha_hist, 1e-6), max(beta_hist, 1e-6))
        fig.add_trace(go.Scatter(
            x=x_range, y=prior_pdf, name="Prior (baseline)",
            line=dict(color="black", dash="dash", width=2)
        ))
    else:
        # Jeffreys prior for reference
        prior_pdf = stats.beta.pdf(x_range, 0.5, 0.5)
        fig.add_trace(go.Scatter(
            x=x_range, y=prior_pdf, name="Jeffreys prior (0.5, 0.5)",
            line=dict(color="black", dash="dash", width=2)
        ))

    a_pdf = stats.beta.pdf(x_range, a_alpha, a_beta)
    b_pdf = stats.beta.pdf(x_range, b_alpha, b_beta)

    fig.add_trace(go.Scatter(x=x_range, y=a_pdf, name="Posterior A", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=x_range, y=b_pdf, name="Posterior B", fill="tozeroy"))

    fig.update_layout(
        xaxis_title="Rate p",
        yaxis_title="Density",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Posterior means with credible intervals")
    fig_bar = go.Figure()
    y_labels = ["A", "B"]
    means = [a_mean, b_mean]
    lows = [a_ci[0], b_ci[0]]
    highs = [a_ci[1], b_ci[1]]

    fig_bar.add_trace(go.Bar(
        y=y_labels,
        x=means,
        orientation="h",
        text=[f"{m:.2%}" for m in means],
        textposition="auto",
        error_x=dict(
            type="data",
            symmetric=False,
            array=[highs[i] - means[i] for i in range(2)],
            arrayminus=[means[i] - lows[i] for i in range(2)],
            visible=True
        )
    ))
    fig_bar.update_layout(xaxis_title="Posterior mean rate", height=280, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Posterior Variance Shrinks as Sample Size Grows")
    st.write("Holding the prior fixed, adding more trials makes the posterior tighter (lower variance).")

    # Simulate variance curve for a range of sample sizes around a chosen base rate
    # Use the observed A rate as a sensible anchor
    base_rate = clicks_a / views_a if views_a > 0 else hist_ctr

    # Use sample sizes on a log-like grid for a nicer curve
    sample_sizes = np.unique(np.round(np.geomspace(10, 50000, 60)).astype(int))

    # Prior for this plot: either historical prior or Jeffreys
    if prior_weight > 0:
        a0, b0 = alpha_hist, beta_hist
    else:
        a0, b0 = 0.5, 0.5

    variances = []
    ci_widths = []

    for n in sample_sizes:
        s = int(round(base_rate * n))
        f = n - s
        a_post = a0 + s
        b_post = b0 + f
        variances.append(stats.beta.var(a_post, b_post))

        lo = stats.beta.ppf(0.025, a_post, b_post)
        hi = stats.beta.ppf(0.975, a_post, b_post)
        ci_widths.append(hi - lo)

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=sample_sizes,
        y=variances,
        mode="lines",
        name="Posterior variance"
    ))
    fig_var.update_layout(
        xaxis_title="Sample size (trials)",
        yaxis_title="Posterior variance",
        height=420
    )
    st.plotly_chart(fig_var, use_container_width=True)

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=sample_sizes,
        y=ci_widths,
        mode="lines",
        name="95% CI width"
    ))
    fig_ci.update_layout(
        xaxis_title="Sample size (trials)",
        yaxis_title="Width of 95% credible interval",
        height=420
    )
    st.plotly_chart(fig_ci, use_container_width=True)

    st.caption(
        "Both curves should fall as sample size increases: uncertainty shrinks with more observations."
    )

with tab3:
    st.subheader("Probability Outcomes from Posterior Draws")
    st.write("Another useful property: probabilities of comparative events can be estimated by Monte Carlo.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("P(B > A)", f"{p_b_gt_a:.1%}")
    with c2:
        st.metric("P(lift > 0)", f"{p_lift_pos:.1%}")
    with c3:
        st.metric(f"P(lift > {mde_percent:.1%})", f"{p_lift_gt_mde:.1%}")

    # Outcome buckets
    big_win = float(np.mean(lift > 0.10))
    meaningful = float(np.mean((lift > mde_percent) & (lift <= 0.10)))
    small_win = float(np.mean((lift > 0) & (lift <= mde_percent)))
    loss = float(np.mean(lift <= 0))

    fig_pie = go.Figure(data=[go.Pie(
        labels=[
            "Big win (>10%)",
            f"Meaningful ({mde_percent:.1%} to 10%)",
            f"Small win (0 to threshold)",
            "Loss (≤0)"
        ],
        values=[big_win, meaningful, small_win, loss],
        hole=0.45
    )])
    fig_pie.update_layout(height=380, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    st.subheader("Raw data (for transparency)")
    st.table(pd.DataFrame({
        "Group": ["A", "B"],
        "Successes": [clicks_a, clicks_b],
        "Trials": [views_a, views_b],
        "Empirical rate": [f"{clicks_a/views_a:.2%}", f"{clicks_b/views_b:.2%}"]
    }))
