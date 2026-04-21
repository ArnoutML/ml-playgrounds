"""Main Streamlit entrypoint for the ML Overfitting Playground."""

from __future__ import annotations
from src.metrics import regression_metrics, classify_fit

import streamlit as st

from src.data import generate_dataset
from src.model import build_polynomial_model, fit_model, predict
from src.plots import make_curve_plot, make_error_bar_chart


st.set_page_config(
    page_title="ML Best Fit Playground",
    page_icon="🧠",
    layout="wide",
)

true_function_name = st.sidebar.selectbox(
    "True function",
    options=[
        "Linear",
        "Quadratic",
        "Cubic",
        "Sine",
        "Sine + Trend",
        "Wavy Nonlinear",
    ],
    index=4,
    help="Choose the hidden data-generating function the model is trying to learn.",
)

st.title("🧠 ML Best Fit Playground")
st.write("Explore how model complexity, noise, and dataset size affect generalization.")

st.sidebar.header("Controls")

true_function_name = st.sidebar.selectbox(
    "True function",
    options=[
        "Linear",
        "Quadratic",
        "Cubic",
        "Sine",
        "Sine + Trend",
        "Wavy Nonlinear",
    ],
    index=4,
    help="Choose the hidden data-generating function the model is trying to learn.",
    key="true_function_selectbox",
)

degree = st.sidebar.slider(
    "Polynomial degree",
    min_value=1,
    max_value=16,
    value=4,
    help="Low degree can underfit. High degree can overfit.",
)

n_samples = st.sidebar.slider(
    "Number of samples",
    min_value=20,
    max_value=100,
    value=60,
    step=10,
)

noise_std = st.sidebar.slider(
    "Noise level",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

test_size = st.sidebar.slider(
    "Validation split",
    min_value=0.1,
    max_value=0.5,
    value=0.35,
    step=0.05,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=1337,
)

x_train, x_val, y_train, y_val = generate_dataset(
    n_samples=n_samples,
    noise_std=noise_std*0.5,
    function_name=true_function_name,
    test_size=test_size,
    random_seed=int(random_seed),
)

model = build_polynomial_model(degree=degree)
model = fit_model(model, x_train, y_train)

y_train_pred = predict(model, x_train)
y_val_pred = predict(model, x_val)

train_metrics = regression_metrics(y_train, y_train_pred)
val_metrics = regression_metrics(y_val, y_val_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train RMSE", f"{train_metrics['rmse']:.3f}")
col2.metric("Validation RMSE", f"{val_metrics['rmse']:.3f}")
col3.metric("Train R²", f"{train_metrics['r2']:.3f}")
col4.metric("Validation R²", f"{val_metrics['r2']:.3f}")

gap = val_metrics["rmse"] - train_metrics["rmse"]
train_rmse = train_metrics["rmse"]
val_rmse = val_metrics["rmse"]

# These thresholds are heuristic, not universal.
# They work reasonably well for this demo dataset.
fit_label, fit_message = classify_fit(
    train_rmse=train_metrics["rmse"],
    val_rmse=val_metrics["rmse"],
    noise_std=noise_std,
)

if fit_label == "underfitting":
    st.error(fit_message)
elif fit_label == "overfitting":
    st.warning(fit_message)
else:
    st.success(fit_message)

left, right = st.columns([2, 1])

with left:
    st.plotly_chart(
        make_curve_plot(
            x_train,
            y_train,
            x_val,
            y_val,
            model,
            function_name=true_function_name,
        ),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        make_error_bar_chart(train_metrics["rmse"], val_metrics["rmse"]),
        use_container_width=True,
    )

with st.expander("What should I look for?"):
    st.markdown(
        """
        - The goal is to find the lowest **validation error** while the model is not overfitting 
        - The **true function** is the hidden pattern that generated the data.
        - Our model does **not** know this function directly — it only sees noisy samples.
        - **Underfitting** happens when the model is too simple to capture the pattern.
        - **Overfitting** happens when the model learns noise or overly specific quirks in the training set.
        """
    )

if fit_label == "underfitting":
    st.markdown("### 🔵 Fit Status: Underfitting")
elif fit_label == "overfitting":
    st.markdown("### 🟠 Fit Status: Overfitting")
else:
    st.markdown("### 🟢 Fit Status: Good Fit")