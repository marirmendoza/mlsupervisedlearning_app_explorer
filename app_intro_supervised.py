import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ============================================================
# INSTITUTIONAL HEADER
# ============================================================

st.set_page_config(page_title="Introduction to Supervised Learning", layout="wide")

st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #007bff;">
    <strong>Machine Learning – Profa. Mariana Recamonde Mendoza</strong><br>
    Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).<br>
    <em>Interactive material developed with generative AI support (Gemini 3.1 Pro and ChatGPT 5.2).</em>
</div>
""", unsafe_allow_html=True)

st.title("What is Supervised Learning?")

st.markdown("""
Supervised Learning is like learning with a teacher. We provide the computer with **examples** (data) and the **correct answers** (labels). 
The computer then tries to find a pattern that connects them, so it can predict answers for new, unseen examples.

Explore the three scenarios below to understand the two main tasks: **Regression** and **Classification**.
""")

# ============================================================
# TABS FOR SCENARIOS
# ============================================================

tab1, tab2, tab3 = st.tabs(["Regression: The Line Hunter", "Classification: The Best Divider", "Going Beyond: The Overfitter"])

# ============================================================
# TAB 1: REGRESSION
# ============================================================

with tab1:
    st.header("Regression: Predicting Numbers")
    st.write("""
    In **Regression**, the goal is to predict a continuous numerical value. 
    Imagine you are trying to predict the height of a plant based on how many days it has been growing.
    """)
    
    # Generate some synthetic data for regression
    np.random.seed(42)
    x_reg = np.linspace(0, 10, 15)
    y_reg = 2.5 * x_reg + 3 + np.random.normal(0, 2, 15)
    
    col_reg_left, col_reg_right = st.columns([1, 2])
    
    with col_reg_left:
        st.subheader("Adjust your Model")
        slope = st.slider("Slope (m) - How steep is the line?", 0.0, 5.0, 1.0, 0.1)
        intercept = st.slider("Intercept (b) - Where does it start?", 0.0, 10.0, 5.0, 0.1)
        
        y_pred_user = slope * x_reg + intercept
        error = np.mean((y_reg - y_pred_user)**2)
        
        st.metric("Total Error", f"{error:.2f}", help="Lower is better!")
        
        if error < 5:
            st.success("Great job! That line fits the points very well.")
        elif error < 15:
            st.info("You are getting close! Try adjusting the slope a bit more.")

        st.markdown("---")
        st.subheader("Predict for a New Point")
        new_x = st.slider("Select a day to predict:", 0.0, 15.0, 12.0)
        new_y = slope * new_x + intercept
        st.write(f"Based on your line, a plant at day **{new_x}** will be **{new_y:.2f} cm** tall.")

    with col_reg_right:
        fig_reg = go.Figure()
        # Data points
        fig_reg.add_trace(go.Scatter(x=x_reg, y=y_reg, mode='markers', name='Actual Data', marker=dict(size=10, color='blue')))
        # User line
        x_line = np.array([0, 15])
        y_line = slope * x_line + intercept
        fig_reg.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Your Model', line=dict(color='red', width=3)))
        
        # New prediction point (unlabeled)
        fig_reg.add_trace(go.Scatter(x=[new_x], y=[new_y], mode='markers', name='New Prediction', 
                                    marker=dict(size=15, color='gray', symbol='x', line=dict(width=2, color='black'))))
        
        fig_reg.update_layout(
            title="Plant Growth vs. Time (Analogy)",
            xaxis_title="Time (Days)",
            yaxis_title="Height (cm)",
            template="plotly_white",
            height=500,
            xaxis=dict(range=[0, 15]),
            yaxis=dict(range=[0, 45])
        )
        st.plotly_chart(fig_reg, use_container_width=True)


# ============================================================
# TAB 2: CLASSIFICATION
# ============================================================

with tab2:
    st.header("Classification: Sorting into Categories")
    st.write("""
    In **Classification**, the goal is to predict a discrete category or 'label'. 
    Imagine you are a machine trying to separate apples (red points) from oranges (blue points) based on their weight and color intensity.
    """)
    
    # Select scenario
    scenario = st.radio("Choose the Scenario:", ["Linearly Separable", "Non-Linearly Separable (Circular)"])
    
    # Generate some synthetic data for classification
    np.random.seed(42)
    X_cls = np.random.rand(30, 2) * 10
    
    if scenario == "Linearly Separable":
        y_cls = (X_cls[:, 0] + X_cls[:, 1] > 10).astype(int)
        st.info("This problem can be perfectly solved with a straight line (a linear model).")
    else:
        # Circular pattern: points inside distance R from center are class 1
        center = np.array([5, 5])
        dist = np.linalg.norm(X_cls - center, axis=1)
        y_cls = (dist < 3.5).astype(int)
        st.warning("⚠️ This problem is **NOT** linearly separable. No matter how you move the straight line, you will always misclassify some points.")

    col_cls_left, col_cls_right = st.columns([1, 2])
    
    with col_cls_left:
        st.subheader("Adjust the Divider")
        angle = st.slider("Rotation Angle", -2.5, 2.5, 0.0, 0.1)
        offset = st.slider("Position Offset", -15.0, 15.0, 5.0, 0.5)
        
        def user_predict(X):
            return (angle * X[:, 0] - X[:, 1] + offset > 0).astype(int)
        
        preds = user_predict(X_cls)
        correct = np.sum(preds == y_cls)
        total = len(y_cls)
        
        st.metric("Success Meter", f"{correct}/{total} Correct", help="Try to separate all Blue from Red points!")
        
        if correct == total:
            st.success("Perfect separation! You found the general pattern.")
        elif correct > total * 0.8:
            st.info("Almost there! A few points are still on the wrong side.")

        st.markdown("---")
        st.subheader("Predict for a New Item")
        c_x1 = st.slider("New Item Feature 1:", 0.0, 10.0, 2.0)
        c_x2 = st.slider("New Item Feature 2:", 0.0, 10.0, 2.0)
        new_point_cls = user_predict(np.array([[c_x1, c_x2]]))[0]
        label = "Blue Class" if new_point_cls == 1 else "Red Class"
        st.write(f"This new item would be classified as: **{label}**")

    with col_cls_right:
        fig_cls = go.Figure()
        # Data points
        colors = ['red' if val == 0 else 'blue' for val in y_cls]
        fig_cls.add_trace(go.Scatter(x=X_cls[:, 0], y=X_cls[:, 1], mode='markers', 
                                    name='Training Items', 
                                    marker=dict(size=12, color=colors, line=dict(width=1, color='black'))))
        
        # User separator line
        x_sep = np.array([0, 10])
        y_sep = angle * x_sep + offset
        fig_cls.add_trace(go.Scatter(x=x_sep, y=y_sep, mode='lines', name='Decision Boundary', line=dict(color='black', width=4, dash='dash')))
        
        # New prediction point (unlabeled)
        fig_cls.add_trace(go.Scatter(x=[c_x1], y=[c_x2], mode='markers', name='New Item', 
                                    marker=dict(size=16, color='gray', symbol='star', line=dict(width=2, color='black'))))
        
        fig_cls.update_layout(
            title="Separating Categories (Apples vs Oranges)",
            xaxis_title="Feature 1 (e.g. Weight)",
            yaxis_title="Feature 2 (e.g. Color Intensity)",
            template="plotly_white",
            height=500,
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10])
        )
        st.plotly_chart(fig_cls, use_container_width=True)


# ============================================================
# TAB 3: OVERFITTING
# ============================================================

with tab3:
    st.header("Scenario 3: The Danger of Fitting Too Much")
    st.write("""
    Machines can sometimes become 'too focused' on the details of the examples they see. 
    This is called **Overfitting**. If we fit the data too precisely, we might fail to learn the underlying 
    **general pattern**, making the model useless for new cases.
    """)
    
    # Generate complex data
    np.random.seed(7)
    X_train = np.sort(np.random.rand(10) * 10)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, 10)
    
    X_test = np.linspace(0, 10, 50)
    y_test = np.sin(X_test)
    
    col3_left, col3_right = st.columns([1, 2])
    
    with col3_left:
        st.subheader("Model Complexity")
        degree = st.slider("Curve Complexity (Polynomial Degree)", 1, 15, 1)
        show_test = st.checkbox("Reveal Hidden Patterns (New Data)", value=False)
        
        # Train model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train.reshape(-1, 1), y_train)
        
        train_pred = model.predict(X_train.reshape(-1, 1))
        test_pred = model.predict(X_test.reshape(-1, 1))
        
        train_error = np.mean((y_train - train_pred)**2)
        test_error = np.mean((y_test - test_pred)**2)
        
        st.metric("Training Error (Visible)", f"{train_error:.4f}")
        if show_test:
            st.metric("Generalization Error (Hidden)", f"{test_error:.4f}")
            
            if degree > 8:
                st.warning("Notice how the curve wiggles to hit every single point? That's overfitting!")
            elif degree < 3:
                st.info("The line is too simple to see the wave. This is underfitting.")


    with col3_right:
        fig_over = go.Figure()
        # Train points
        fig_over.add_trace(go.Scatter(x=X_train, y=y_train, mode='markers', name='Visible Examples', marker=dict(size=12, color='blue')))
        
        # The true pattern (hidden)
        if show_test:
            fig_over.add_trace(go.Scatter(x=X_test, y=y_test, mode='lines', name='Actual Pattern', line=dict(color='green', dash='dot')))
            fig_over.add_trace(go.Scatter(x=X_test[::5], y=y_test[::5] + np.random.normal(0, 0.1, 10), mode='markers', 
                                         name='New Samples', marker=dict(symbol='x', color='green', size=8)))
        
        # Model curve
        x_range = np.linspace(0, 10, 200)
        y_curve = model.predict(x_range.reshape(-1, 1))
        fig_over.add_trace(go.Scatter(x=x_range, y=y_curve, mode='lines', name='The Model', line=dict(color='red', width=3)))
        
        fig_over.update_layout(
            title="Complexity vs. Generalization",
            template="plotly_white",
            height=500,
            yaxis=dict(range=[-2, 2])
        )
        st.plotly_chart(fig_over, use_container_width=True)

    st.info("""
    **The Lesson**: A perfect score on visible data doesn't mean the machine learned correctly. 
    A model that is too complex often learns the 'noise' (random errors) instead of the actual signal.
    """)
