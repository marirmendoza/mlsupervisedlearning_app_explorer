# Supervised Learning Explorer — Understanding Regression and Classification

This interactive application, built with **Streamlit** and **Python**, allows you to explore the fundamentals of **Supervised Learning** in a visual and intuitive way.

Supervised Learning is the branch of Machine Learning where models learn from labeled examples—data paired with correct answers. By observing these patterns, the model becomes capable of predicting answers for new, unseen cases.

This app breaks down the two primary types of supervised tasks: **Regression** (predicting numbers) and **Classification** (assigning categories). It also introduces the critical concept of **Overfitting**, where a model becomes "too smart for its own good" by memorizing noise instead of learning the general signal.

This app was developed by **Prof. Mariana Recamonde Mendoza** as supporting material for the **Machine Learning** course taught at the **Institute of Informatics — Federal University of Rio Grande do Sul (UFRGS)**.

🔗 [https://mlsupervisedlearning-app-explorer.streamlit.app](https://mlsupervisedlearning-app-explorer.streamlit.app)

---

## App Goal

The goal of this explorer is to demystify how models "fit" to data. It emphasizes the human-in-the-loop experience, allowing users to manually act as the learning algorithm.

This explorer allows you to visualize:

- **The Line Hunter (Regression)**: How adjusting slope and intercept minimizes error when predicting continuous values.
- **The Great Divider (Classification)**: How a decision boundary separates different classes and why some problems (non-linear) can't be solved with a simple straight line.
- **The Overfitter**: The trade-off between model complexity and generalization, and why a "perfect" fit on training data can be misleading.
- **Reactive Predictions**: Instantly see how new, unlabeled data points are handled by your manual model.

---

## App Overview

The application is organized into three interactive scenarios:

1. **Regression: The Line Hunter**: Predict the height of a plant over time by manually adjusting a linear model to minimize the "Total Error".
2. **Classification: The Great Divider**: Categorize items by rotating and positioning a decision boundary. Experiment with **Linearly Separable** vs. **Circular** data distributions.
3. **Scenario 3: The Overfitter**: Adjust the complexity of a polynomial curve and witness how "hidden patterns" reveal the dangers of fitting too closely to random noise.

---

## Interactive Controls

In each scenario, you can interact with:

- **Model Parameters**: Sliders to control Slope, Intercept, Rotation, and Position of your models.
- **Scenario Toggle**: Switch between different data distributions in classification to see the limits of linear models.
- **Prediction Sliders**: Move a "new item" (marker 'x' or 'star') across the features space to see how the model generalizes in real-time.
- **Complexity Slider**: Increase the degree of the polynomial to see the model transition from Underfitting to Overfitting.

---

## Educational Highlights

The app includes three main didactic areas:

### 1️⃣ Regression vs. Classification

This highlights the fundamental difference between predicting a **continuous value** (like height or price) and a **discrete category** (like "Apple" vs. "Orange").

### 2️⃣ Linear Separability

By switching between scenarios in the Classification tab, you can observe:
- How a straight line easily separates some data.
- How complex, non-linear patterns (like a circle) are impossible for a simple linear divider to solve perfectly.

### 3️⃣ Generalization and Noise

In the Overfitting scenario, the app demonstrates:
- **Underfitting**: When the model is too simple to capture the trend.
- **Overfitting**: When the model is so complex it captures random "wiggles" in the training data, failing on new test samples.
- The importance of **Generalization Error** as the true metric of success.

---

## Credits

**Author:** Profa. Mariana Recamonde Mendoza. 

🔗 [Personal website.](https://www.inf.ufrgs.br/~mrmendoza/)

📍 [Institute of Informatics](https://www.inf.ufrgs.br/site/) - Federal University of Rio Grande do Sul (UFRGS), Porto Alegre - RS, Brazil

---
## Notes
*The code was developed with the support of Generative AI (Gemini 3.1 and ChatGPT 5.2).*
