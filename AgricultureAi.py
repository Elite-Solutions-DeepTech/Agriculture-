import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import io  # Import io module

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

page_bg_img = '''
<style>
body {
background-color: #f0f0f0;  /* Change this to your desired background color */
}
</style>
'''

# Load dataset
df = pd.read_csv('/mount/src/agriculture_ai/Crop_recommendation.csv')

st.title('Crop Recommendation App')

# Display dataset
# st.write("Dataset preview:")
# st.write(df.head(6))

# Dataset information
# st.write("Dataset information:")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
# st.text(s)

# Check for null values
# st.write("Check for null values:")
# st.write(df.isnull().sum())

# Label value counts
# st.write("Label value counts:")
# st.write(df['label'].value_counts())

# Split dataset into features and target
x = df.drop('label', axis=1)
y = df['label']

# Split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

# Train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
y_pred1 = logistic_model.predict(x_test)
logistic_reg_acc = accuracy_score(y_test, y_pred1)


# Train Decision Tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)
y_pred3 = decision_tree_model.predict(x_test)
decision_acc = accuracy_score(y_test, y_pred3)


# Train Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(x_train, y_train)
y_pred4 = random_forest_model.predict(x_test)
random_acc = accuracy_score(y_test, y_pred4)


# Save the Decision Tree model
joblib.dump(decision_tree_model, 'crop_app')

# Load the model
app = joblib.load('crop_app')

# User input for prediction
st.write("Predict crop recommendation based on user input:")
N = st.number_input('Nitrogen', min_value=0)
P = st.number_input('Phosphorus', min_value=0)
K = st.number_input('Potassium', min_value=0)
temperature = st.number_input('Temperature', min_value=0.0)
humidity = st.number_input('Humidity', min_value=0.0)
ph = st.number_input('pH', min_value=0.0)
rainfall = st.number_input('Rainfall', min_value=0.0)

if st.button('Predict'):
    arr = [[N, P, K, temperature, humidity, ph, rainfall]]
    y_pred5 = app.predict(arr)
    st.write(f"Recommended Crop: {y_pred5[0]}")

# Plotting graphs
def plot_graph(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='black', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)
    st.pyplot(plt)

# Logistic Regression graph
st.write("Logistic Regression: Actual vs Predicted")
plot_graph(y_test, y_pred1, 'Actual vs Predicted (Logistic Regression)')

# Decision Tree graph
st.write("Decision Tree: Actual vs Predicted")
plot_graph(y_test, y_pred3, 'Actual vs Predicted (Decision Tree)')

# Random Forest graph
st.write("Random Forest: Actual vs Predicted")
plot_graph(y_test, y_pred4, 'Actual vs Predicted (Random Forest)')

st.write(f"Logistic Regression accuracy: {logistic_reg_acc}")
st.write(f"Decision Tree accuracy: {decision_acc}")
st.write(f"Random Forest accuracy: {random_acc}")
