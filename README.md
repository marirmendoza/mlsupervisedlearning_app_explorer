# Supervised Learning Explorer — Understanding Regression and Classification

This interactive application, built with **Streamlit** and **Python**, allows you to explore the fundamentals of **Supervised Learning** in a visual and intuitive way.

Supervised Learning is the branch of Machine Learning (ML) where models learn from labeled examples, that is, each example has input features and it is paired with a correct expected answer.  By observing these patterns and fitting the model, the goal is to become capable of predicting answers for new, unseen cases. The goal is simple:

> Learn a pattern from known examples and use it to make predictions on new, unseen data.

This app breaks down the two primary types of supervised tasks: **Regression** (predicting numbers) and **Classification** (predicting categories). It also introduces the critical concept of **Overfitting** in ML, where a model becomes "too smart for its own good" by memorizing noise instead of learning the general signal.

This app was developed by **Prof. Mariana Recamonde Mendoza** as supporting material for the **Machine Learning** course taught at the **Institute of Informatics — Federal University of Rio Grande do Sul (UFRGS)**.

🔗 [https://inf-supervisedml-app-explorer.streamlit.app](https://inf-supervisedml-app-explorer.streamlit.app)

---

## App Goal

The goal of this explorer is to demystify how models "fit" to data. It emphasizes the human-in-the-loop experience, allowing users to manually act as the learning algorithm, by adjusting its parameters until the model becomes able to solve the task with a satisfactory performance.

This explorer allows you to visualize:

- **The Line Hunter (Regression)**: How adjusting slope and intercept minimizes error when predicting continuous values, assuming a linear model. 
- **The Great Divider (Classification)**: How a decision boundary separates different classes and why some problems (non-linear) can't be solved with a simple straight line.
- **The Overfitter**: The trade-off between model complexity and generalization, and why a "perfect" fit on training data can sometimes be misleading.
- **Reactive Predictions**: Instantly see how new, unlabeled data points are handled by your manually fitted model.


---
## Who Is This For?

This app is designed for:

- Students taking their first ML course
- Beginners who want a visual and intuitive explanation
- Instructors who want an interactive teaching tool

No advanced math is required to explore the concepts.

---

## App Overview

The application is organized into three interactive scenarios:

1. **Regression: The Line Hunter**: Predict the height of a plant over time by manually adjusting a linear model to minimize the "Total Error".
2. **Classification: The Best Divider**: Categorize items by rotating and positioning a decision boundary. Experiment with **Linearly Separable** vs. **Circular** data distributions.
3. **Going beyond: The Overfitter**: Adjust the complexity of a polynomial curve and witness how "hidden patterns" reveal the dangers of fitting too closely to random noise.

---

## Interactive Controls

In each scenario, you can interact with:

- **Model Parameters**: Sliders to control Slope, Intercept, Rotation, and Position of your regression or classification models.
- **Scenario Toggle**: Switch between different data distributions in classification to see the limits of linear models.
- **Prediction Sliders**: Move a "new item" (marker 'x' or 'star') across the features space to see how the model generalizes for new data.
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
- **Underfitting**: A model underfits when it is **too simple** to capture the true pattern in the data. It performs poorly on training data and on new data, because it misses important structure.
- **Overfitting**: A model overfits when it is **too complex** and starts memorizing random noise or fuctuations (“wiggles”) instead of learning the true pattern. It performs very well on training data, but poorly on new data.
- The importance of **Generalization Error** as the true metric of success.

---

## Credits

**Author:** Profa. Mariana Recamonde Mendoza. 

🔗 [Personal website.](https://www.inf.ufrgs.br/~mrmendoza/)

📍 [Institute of Informatics](https://www.inf.ufrgs.br/site/) - Federal University of Rio Grande do Sul (UFRGS), Porto Alegre - RS, Brazil

---
## Notes
*The code was developed with the support of Generative AI (Gemini 3.1 and ChatGPT 5.2).*
