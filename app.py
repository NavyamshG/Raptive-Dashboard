!pip install streamlit
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Raptive Stats Demo", layout="wide")

st.title("📊 Statistical Demo: The Central Limit Theorem")
st.markdown("""
The **Central Limit Theorem (CLT)** states that if you take enough samples from *any* distribution, the distribution of the **sample means** will eventually look like a Normal (Bell Curve) distribution.
""")

# Sidebar settings
st.sidebar.header("Simulation Parameters")
dist_type = st.sidebar.selectbox("1. Choose a 'Messy' Population Distribution",
                                 ["Exponential (Heavy Skew)", "Uniform (Flat)"])
sample_size = st.sidebar.slider("2. Select Sample Size (n)", 1, 100, 30)
num_samples = st.sidebar.slider("3. Number of Samples to draw", 100, 5000, 1000)

# Generate Data
if dist_type == "Exponential (Heavy Skew)":
    pop = np.random.exponential(scale=2, size=10000)
else:
    pop = np.random.uniform(low=0, high=10, size=10000)

# Calculate Sample Means
means = [np.mean(np.random.choice(pop, size=sample_size)) for _ in range(num_samples)]

# Visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("The Raw Population")
    fig1, ax1 = plt.subplots()
    sns.histplot(pop, kde=True, color="#3498db", ax=ax1)
    ax1.set_title(f"Original {dist_type} Data")
    st.pyplot(fig1)

with col2:
    st.subheader("The Distribution of Sample Means")
    fig2, ax2 = plt.subplots()
    sns.histplot(means, kde=True, color="#e74c3c", ax=ax2)
    ax2.set_title(f"Means of {num_samples} Samples (n={sample_size})")
    st.pyplot(fig2)

st.success(f"**Insight:** Even though the first chart is {dist_type.lower()}, the second chart becomes a Normal 'Bell Curve' as you increase the sample size!")
