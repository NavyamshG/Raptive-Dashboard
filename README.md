# Raptive-Dashboard
This repo is for Raptive take home.
# 📊 CTR Inference Lab

A high-impact **Streamlit** laboratory designed for evaluating the robustness of A/B test results. This tool visualizes how different distributional assumptions and statistical "lenses" (Z-test, T-test, and Fisher’s Exact) impact the validity of conversion rate optimization (CRO) decisions.

## 🚀 Key Features

* **Multi-Estimator Engine**: Simultaneously compares p-values from:
    * **Z-Test**: The standard normal approximation for large samples.
    * **Welch’s T-Test**: A robust method for unequal variances and sample sizes.
    * **Fisher’s Exact Test**: The "gold standard" for small samples, calculating exact combinatorial probabilities.
* **Advanced Confidence Intervals**: Compares the traditional **Wald (Binomial) Interval** against the **Newcombe-Wilson Interval** to demonstrate how bound estimations shift under different frameworks.
* **The "Peeking" Simulator**: A live Monte Carlo simulation showing how checking your results before a test concludes exponentially inflates your False Positive Rate (Type I Error).
* **High-Fidelity Visuals**: Dynamic Plotly charts for distribution overlaps, p-value comparison bars, and "P-value journey" tracking.

## 🧪 The "Magic" Scenario
This lab is designed to reveal "Conflicts of Significance." 
**Try these inputs:**
- **Control**: 20 Clicks / 200 Views
- **Variant**: 35 Clicks / 200 Views
- **Significance ($\alpha$)**: 0.04

You will observe a scenario where the Z and T tests declare a **Winner**, while Fisher’s Exact remains **Non-Significant**. This demonstrates the danger of relying on normal approximations in marginal volume scenarios.



## 🛠️ Installation & Usage

### 1. Requirements
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install streamlit numpy scipy plotly
