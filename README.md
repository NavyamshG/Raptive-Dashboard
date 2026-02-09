# 📊 CTR Inference Lab

A high-impact **Streamlit** laboratory designed for evaluating the robustness of A/B test results. This tool visualizes how different distributional assumptions and statistical lenses (Z-test, T-test, and Fisher’s Exact) impact conversion rate optimization (CRO) decisions.

---

## 🚀 Key Features

- **Multi-Estimator Engine**
  - Z-Test (normal approximation)
  - Welch’s T-Test
  - Fisher’s Exact Test

- **Advanced Confidence Intervals**
  - Wald Interval
  - Newcombe-Wilson Interval

- **Peeking Simulator**
  Demonstrates false positive inflation from repeated interim checks.

- **Interactive Visualizations**
  Built using Plotly for intuitive inference comparison.

---

## 🧪 The "Magic" Scenario

Try:

- **Control:** 20 / 200  
- **Variant:** 35 / 200  
- **α:** 0.04  

Z and T may show significance while Fisher does not.

This highlights approximation risk in moderate sample experiments.

---

## 🖼️ Preview


<img src="image1.png" alt="Raptive-Dashboard" width="800"/>

<img src="image2.png" alt="Raptive-Dashboard" width="800"/>

<img src="image3.png" alt="Raptive-Dashboard" width="800"/>

<img src="image4.png" alt="Raptive-Dashboard" width="800"/>

## 🛠 Installation & Usage

### 1 Install Dependencies

Make sure Python 3.8+ is installed.

```bash
pip install streamlit pandas numpy matplotlib seaborn scipy plotly

### 2. Install Dependencies

streamlit run app.py
