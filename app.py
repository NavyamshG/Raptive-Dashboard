import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Raptive Decision Engine", layout="wide")
st.title("🏆 Executive A/B Decision Engine")
st.caption("Bayesian A/B decisioning with optional historical baseline")

rng = np.random.default_rng()

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("📊 Experiment Inputs")

    with st.expander("Current Test Data", expanded=True):
        col_a, col_b = st.columns(2)

        clicks_a = col_a.number_input("Control Clicks", value=100, min_value=0, step=1)
        views_a = col_a.number_input("Control Views", value=1000, min_value=0, step=1)

        clicks_b = col_b.number_input("Variant Clicks", value=130, min_value=0, step=1)
        views_b = col_b.number_input("Variant Views", value=1100, min_value=0, step=1)

    with st.expander("⚙️ Historical Baseline (Prior)", expanded=True):
        st.write("Use business knowledge as a prior for the baseline rate.")
        hist_ctr = st.slider("Historical CTR (%)", 0.01, 30.0, 10.0) / 100.0

        prior_weight = st.number_input(
            "Prior Strength (effective trials)",
            value=500,
            min_value=0,
            step=50,
            help="Higher = prior dominates more. Set 0 to disable prior."
        )

        prior_mode = st.radio(
            "Prior Mode",
            options=["Shared prior (A & B)", "Baseline-only prior (A only)"],
            help="Shared prior anchors both arms equally. Baseline-only anchors control; variant uses a weak prior."
        )

        weak_prior_weight = st.number_input(
            "Variant weak prior weight (only for Baseline-only mode)",
            value=2,
            min_value=0,
            step=1,
            help="Small stabilizer for B when not sharing the historical prior."
        )

    with st.expander("🎯 Decision Thresholds", expanded=True):
        mde_percent = st.slider(
            "Practical threshold (MDE) %",
            0.0, 10.0, 1.0, 0.1,
            help="Minimum lift that is worth shipping."
        ) / 100.0

        deploy_conf = st.slider(
            "Deploy when P(lift > MDE) ≥",
            0.50, 0.99, 0.90, 0.01
        )

        loss_cap = st.slider(
            "Loss risk cap P(lift < 0) ≤",
            0.01, 0.50, 0.10, 0.01
        )

    with st.expander("💰 Financial Model", expanded=True):
        traffic_unit = st.selectbox(
            "Traffic unit",
            ["Monthly Impressions", "Monthly Sessions", "Monthly Visitors"],
            index=0
        )

        monthly_traffic = st.number_input(f"{traffic_unit}", value=100000, min_value=0, step=1000)

        # If user chooses visitors, allow views per visitor
        views_per_unit = st.number_input(
            "Views per unit (multiplier)",
            value=1.0,
            min_value=0.0,
            step=0.1,
            help="If you input Visitors/Sessions but CTR is defined on views/impressions, set the average views per unit."
        )

        val_per_click = st.number_input("Value per Click ($)", value=50.0, min_value=0.0, step=1.0)

    with st.expander("🧪 Simulation", expanded=False):
        n_sims = st.slider("Posterior draws", 2000, 50000, 20000, 1000)
        seed = st.number_input("Seed", value=42, min_value=0, step=1)


# -------------------------
# VALIDATION
# -------------------------
def validate(clicks: int, views: int, label: str) -> None:
    if views < clicks:
        st.error(f"{label}: Views must be ≥ Clicks (got clicks={clicks}, views={views}).")
        st.stop()
    if views == 0:
        st.error(f"{label}: Views must be > 0 to estimate a rate.")
        st.stop()

validate(clicks_a, views_a, "Control")
validate(clicks_b, views_b, "Variant")

# -------------------------
# BAYESIAN MODEL
# -------------------------
# Beta prior from historical CTR
# Guard tiny values for stability when prior_weight=0
alpha_hist = max(prior_weight * hist_ctr, 0.0)
beta_hist = max(prior_weight * (1.0 - hist_ctr), 0.0)

def beta_params_for_arm(clicks, views, arm: str):
    """
    Returns (alpha, beta) posterior params.
    """
    if prior_weight == 0:
        # Jeffreys prior (0.5, 0.5) is a nice default
        a0, b0 = 0.5, 0.5
        return clicks + a0, (views - clicks) + b0

    if prior_mode == "Shared prior (A & B)":
        return clicks + alpha_hist, (views - clicks) + beta_hist

    # Baseline-only prior (A only): historical prior anchors control; variant gets weak stabilizer prior
    if arm == "A":
        return clicks + alpha_hist, (views - clicks) + beta_hist
    else:
        a0 = max(weak_prior_weight * hist_ctr, 0.0) + 0.5
        b0 = max(weak_prior_weight * (1.0 - hist_ctr), 0.0) + 0.5
        return clicks + a0, (views - clicks) + b0


# Simulation
rng = np.random.default_rng(int(seed))

a_alpha, a_beta = beta_params_for_arm(clicks_a, views_a, "A")
b_alpha, b_beta = beta_params_for_arm(clicks_b, views_b, "B")

sim_a = rng.beta(a_alpha, a_beta, int(n_sims))
sim_b = rng.beta(b_alpha, b_beta, int(n_sims))

# Derived quantities
lift = (sim_b - sim_a) / np.clip(sim_a, 1e-9, None)

p_b_gt_a = float(np.mean(sim_b > sim_a))
p_lift_pos = float(np.mean(lift > 0))
p_lift_gt_mde = float(np.mean(lift > mde_percent))
p_loss = float(np.mean(lift < 0))

# Credible intervals
def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))

a_ci = ci(sim_a)
b_ci = ci(sim_b)
lift_ci = ci(lift)

# Central tendency (use median for lift)
lift_med = float(np.median(lift))
lift_mean = float(np.mean(lift))

# -------------------------
# FINANCIALS FROM SIMULATION (EV + RISK)
# -------------------------
annual_traffic_units = float(monthly_traffic) * 12.0
annual_views = annual_traffic_units * float(views_per_unit)

# Revenue under each posterior draw (consistent units)
rev_a = sim_a * annual_views * float(val_per_click)
rev_b = sim_b * annual_views * float(val_per_click)
delta_rev = rev_b - rev_a

delta_ev = float(np.mean(delta_rev))
delta_p05, delta_p95 = ci(delta_rev, 5, 95)
p_neg_delta = float(np.mean(delta_rev < 0))

# Simple baseline (for display only): empirical control CTR * traffic
emp_ctr_a = clicks_a / views_a
emp_ctr_b = clicks_b / views_b
current_annual_rev = emp_ctr_a * annual_views * float(val_per_click)

# Recommendation logic
if (p_lift_gt_mde >= deploy_conf) and (p_loss <= loss_cap):
    rec = "DEPLOY"
    rec_color = "green"
elif (p_loss > 0.35) and (p_lift_gt_mde < 0.5):
    rec = "STOP"
    rec_color = "red"
else:
    rec = "WAIT"
    rec_color = "orange"


# -------------------------
# TOP KPI STRIP
# -------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("P(Variant > Control)", f"{p_b_gt_a:.1%}")
    st.caption("Win probability")

with kpi2:
    st.metric("P(Lift > MDE)", f"{p_lift_gt_mde:.1%}")
    st.caption(f"Meaningful lift (>{mde_percent:.1%})")

with kpi3:
    st.metric("Median Lift", f"{lift_med:+.2%}")
    st.caption(f"95% CI: [{lift_ci[0]:+.2%}, {lift_ci[1]:+.2%}]")

with kpi4:
    st.markdown("**Recommendation**")
    st.markdown(
        f"<h1 style='color:{rec_color}; margin-top:-15px;'>{rec}</h1>",
        unsafe_allow_html=True
    )
    st.caption("Decision rule uses meaningful lift + loss risk")

st.divider()

# -------------------------
# TABS
# -------------------------
tab1, tab2 = st.tabs(["📈 Bayesian Evidence", "💰 Business Impact"])

with tab1:
    st.subheader("Prior vs Posterior (Control and Variant)")
    st.write("This shows how the experiment updates your baseline belief and the uncertainty for each arm.")

    x_max = max(sim_a.max(), sim_b.max(), hist_ctr) * 1.5
    x_max = min(x_max, 1.0)
    x_range = np.linspace(0, max(0.02, x_max), 600)

    fig_evol = go.Figure()

    # Plot historical prior only if enabled
    if prior_weight > 0:
        prior_pdf = stats.beta.pdf(x_range, max(alpha_hist, 1e-6), max(beta_hist, 1e-6))
        fig_evol.add_trace(go.Scatter(
            x=x_range, y=prior_pdf, name="Historical prior (baseline)",
            line=dict(color="black", dash="dash", width=2)
        ))

    control_pdf = stats.beta.pdf(x_range, a_alpha, a_beta)
    variant_pdf = stats.beta.pdf(x_range, b_alpha, b_beta)

    fig_evol.add_trace(go.Scatter(
        x=x_range, y=control_pdf, name="Control posterior",
        fill="tozeroy"
    ))
    fig_evol.add_trace(go.Scatter(
        x=x_range, y=variant_pdf, name="Variant posterior",
        fill="tozeroy"
    ))

    fig_evol.update_layout(
        xaxis_title="CTR / Conversion rate",
        yaxis_title="Density",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    st.divider()

    left, right = st.columns([1.4, 1.0])

    with left:
        st.subheader("Posterior Means with 95% Credible Intervals")

        a_mean = float(np.mean(sim_a))
        b_mean = float(np.mean(sim_b))

        y_labels = ["Control", "Variant"]
        means = [a_mean, b_mean]
        ci_low = [a_ci[0], b_ci[0]]
        ci_high = [a_ci[1], b_ci[1]]

        # Plotly horizontal bar with asymmetric error bars
        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            y=y_labels,
            x=means,
            orientation="h",
            text=[f"{m:.2%}" for m in means],
            textposition="auto",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ci_high[i] - means[i] for i in range(2)],
                arrayminus=[means[i] - ci_low[i] for i in range(2)],
                visible=True
            )
        ))

        fig_bar.update_layout(
            xaxis_title="CTR / Conversion rate",
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(f"Control 95% CI: [{a_ci[0]:.2%}, {a_ci[1]:.2%}] | Variant 95% CI: [{b_ci[0]:.2%}, {b_ci[1]:.2%}]")

    with right:
        st.subheader("Outcome Breakdown (Lift)")
        big_win = float(np.mean(lift > 0.10))
        meaningful = float(np.mean((lift > mde_percent) & (lift <= 0.10)))
        small_win = float(np.mean((lift > 0) & (lift <= mde_percent)))
        loss = float(np.mean(lift <= 0))

        fig_pie = go.Figure(data=[go.Pie(
            labels=[
                "Big win (>10%)",
                f"Meaningful ({mde_percent:.1%} to 10%)",
                f"Small win (0 to {mde_percent:.1%})",
                "Loss (≤0)"
            ],
            values=[big_win, meaningful, small_win, loss],
            hole=0.45
        )])
        fig_pie.update_layout(height=360, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    st.subheader("Why this recommendation")
    bullets = []
    bullets.append(f"- **P(lift > MDE)** = **{p_lift_gt_mde:.1%}** (threshold: {deploy_conf:.0%})")
    bullets.append(f"- **P(loss)** = **{p_loss:.1%}** (cap: {loss_cap:.0%})")
    bullets.append(f"- Median lift = **{lift_med:+.2%}** (95% CI: [{lift_ci[0]:+.2%}, {lift_ci[1]:+.2%}])")
    bullets.append(f"- Expected annual impact (EV) = **${delta_ev:,.0f}**; 5th–95th = **[${delta_p05:,.0f}, ${delta_p95:,.0f}]**")
    st.markdown("\n".join(bullets))

with tab2:
    st.subheader("Annual Financial Impact (Simulated)")

    impact_df = pd.DataFrame({
        "Metric": [
            "Traffic unit",
            "Monthly traffic input",
            "Views per unit",
            "Annual views modeled",
            "Control CTR (empirical)",
            "Variant CTR (empirical)",
            "Expected annual revenue (Control, empirical baseline)",
            "Expected Δ annual revenue (EV)",
            "5th–95th percentile Δ annual revenue",
            "P(Δ revenue < 0)"
        ],
        "Value": [
            traffic_unit,
            f"{monthly_traffic:,.0f}",
            f"{views_per_unit:,.2f}",
            f"{annual_views:,.0f}",
            f"{emp_ctr_a:.2%}",
            f"{emp_ctr_b:.2%}",
            f"${current_annual_rev:,.0f}",
            f"${delta_ev:,.0f}",
            f"[${delta_p05:,.0f}, ${delta_p95:,.0f}]",
            f"{p_neg_delta:.1%}"
        ]
    })

    st.table(impact_df)

    st.info(
        f"**Insight:** Based on your model and current data, the **expected** annual lift is "
        f"**${delta_ev:,.0f}** with a 5th–95th range of **[${delta_p05:,.0f}, ${delta_p95:,.0f}]**."
    )

    st.subheader("Assumptions (make these explicit)")
    st.markdown(
        "- CTR is modeled as a Bernoulli rate over **views** (impressions).\n"
        f"- Your traffic input is **{traffic_unit}** and is converted into annual views via **views per unit = {views_per_unit:.2f}**.\n"
        f"- Value per click = **${val_per_click:,.2f}**.\n"
        f"- Prior strength = **{prior_weight}** effective trials (mode: **{prior_mode}**)."
    )
