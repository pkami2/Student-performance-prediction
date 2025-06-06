
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the dataset using relative path
df = pd.read_csv("student-mat.csv", sep=';')

# Features to use (excluding school, including G1, G2)
features = ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
            'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
            'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

# Data Preprocessing
def preprocess_data(df, fit=True, le_dict=None):
    data = df.copy()
    
    # Define grade categories: Low (0-9), Medium (10-14), High (15-20)
    if 'G3' in data.columns:
        data['G3'] = pd.cut(data['G3'], bins=[-1, 9, 14, 20], labels=[0, 1, 2])  # 0: Low, 1: Medium, 2: High
    
    # Select only relevant features
    if 'G3' in data.columns:
        data = data[features + ['G3']]
    else:
        data = data[features]
    
    # Encode categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    if fit:
        le_dict = {}
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            le_dict[col] = le
    else:
        for col in categorical_columns:
            data[col] = le_dict[col].transform(data[col])
    
    if 'G3' in data.columns:
        X = data.drop(['G3'], axis=1)
        y = data['G3']
        return X, y, le_dict
    return data, le_dict

# Dynamic Weighted Hybrid Ensemble Classifier
class HybridEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_n_estimators=100, nn_hidden_layers=(100, 50), max_iter=500):
        self.rf_n_estimators = rf_n_estimators
        self.nn_hidden_layers = nn_hidden_layers
        self.max_iter = max_iter
        self.rf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42)
        self.xgb = xgb.XGBClassifier(random_state=42)
        self.nn = MLPClassifier(hidden_layer_sizes=nn_hidden_layers, max_iter=max_iter, random_state=42)
        self.scaler = StandardScaler()
        self.rf_weight = 0.4  # Default weights
        self.xgb_weight = 0.3
        self.nn_weight = 0.3

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.rf.fit(X_scaled, y)
        self.xgb.fit(X_scaled, y)
        self.nn.fit(X_scaled, y)
        # Dynamic weighting based on G1 and G2 importance
        g1_idx = X.columns.get_loc('G1')
        g2_idx = X.columns.get_loc('G2')
        g1_g2_importance = self.rf.feature_importances_[g1_idx] + self.rf.feature_importances_[g2_idx]
        if g1_g2_importance > 0.5:  # If G1+G2 dominate
            self.rf_weight = 0.25
            self.xgb_weight = 0.25
            self.nn_weight = 0.50  # Boost NN for grade patterns
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        rf_pred = self.rf.predict_proba(X_scaled)
        xgb_pred = self.xgb.predict_proba(X_scaled)
        nn_pred = self.nn.predict_proba(X_scaled)
        final_pred = (self.rf_weight * rf_pred + self.xgb_weight * xgb_pred + self.nn_weight * nn_pred)
        return np.argmax(final_pred, axis=1)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        rf_pred = self.rf.predict_proba(X_scaled)
        xgb_pred = self.xgb.predict_proba(X_scaled)
        nn_pred = self.nn.predict_proba(X_scaled)
        return (self.rf_weight * rf_pred + self.xgb_weight * xgb_pred + self.nn_weight * nn_pred)
    
    def get_params(self, deep=True):
        return {
            'rf_n_estimators': self.rf_n_estimators,
            'nn_hidden_layers': self.nn_hidden_layers,
            'max_iter': self.max_iter
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.rf = RandomForestClassifier(n_estimators=self.rf_n_estimators, random_state=42)
        self.xgb = xgb.XGBClassifier(random_state=42)
        self.nn = MLPClassifier(hidden_layer_sizes=self.nn_hidden_layers, max_iter=self.max_iter, random_state=42)
        return self

# Train and evaluate models
X, y, le_dict = preprocess_data(df, fit=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hybrid Ensemble Model
hybrid_model = HybridEnsembleClassifier()
hybrid_model.fit(X_train, y_train)
hybrid_pred = hybrid_model.predict(X_test)
hybrid_accuracy = accuracy_score(y_test, hybrid_pred)
hybrid_report = classification_report(y_test, hybrid_pred, target_names=['Low', 'Medium', 'High'])



# Print evaluation results for paper
print("Hybrid Ensemble Model Performance:")
print(f"Accuracy: {hybrid_accuracy:.2f}")
print(hybrid_report)


# Streamlit App
def main():
    st.set_page_config(page_title="Student Performance Prediction", layout="wide")
    st.title("Student Performance Prediction")
    st.write("Enter student details to predict their performance (Low: 0-9, Medium: 10-14, High: 15-20)")

    # Input form in sidebar
    with st.sidebar:
        st.header("Student Details")
        
        # Categorical inputs
        sex = st.selectbox("Sex", ["F", "M"])
        address = st.selectbox("Address Type", ["U", "R"])
        famsize = st.selectbox("Family Size", ["LE3", "GT3"])
        pstatus = st.selectbox("Parent Status", ["T", "A"])
        mjob = st.selectbox("Mother's Job", ["at_home", "health", "other", "services", "teacher"])
        fjob = st.selectbox("Father's Job", ["at_home", "health", "other", "services", "teacher"])
        reason = st.selectbox("Reason for School", ["course", "home", "reputation", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        
        # Numeric inputs
        age = st.slider("Age", 15, 22, 18)
        medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
        fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
        traveltime = st.slider("Travel Time (1-4)", 1, 4, 2)
        studytime = st.slider("Study Time (1-4)", 1, 4, 2)
        failures = st.slider("Past Failures (0-4)", 0, 4, 0)
        famrel = st.slider("Family Relations (1-5)", 1, 5, 4)
        freetime = st.slider("Free Time (1-5)", 1, 5, 3)
        goout = st.slider("Going Out (1-5)", 1, 5, 3)
        dalc = st.slider("Workday Alcohol (1-5)", 1, 5, 1)
        walc = st.slider("Weekend Alcohol (1-5)", 1, 5, 1)
        health = st.slider("Health (1-5)", 1, 5, 3)
        absences = st.slider("Absences (0-93)", 0, 93, 6)
        g1 = st.slider("First Period Grade (0-20)", 0, 20, 10)
        g2 = st.slider("Second Period Grade (0-20)", 0, 20, 10)
        
        # Binary inputs
        schoolsup = st.checkbox("School Support")
        famsup = st.checkbox("Family Support")
        paid = st.checkbox("Paid Classes")
        activities = st.checkbox("Activities")
        nursery = st.checkbox("Nursery")
        higher = st.checkbox("Higher Education")
        internet = st.checkbox("Internet")
        romantic = st.checkbox("Romantic Relationship")
        
        predict_button = st.button("Predict Performance")

    # Prediction logic
    if predict_button:
        # Create input dataframe with exact feature names and order
        input_data = pd.DataFrame({
            'sex': [sex], 'age': [age], 'address': [address], 'famsize': [famsize],
            'Pstatus': [pstatus], 'Medu': [medu], 'Fedu': [fedu], 'Mjob': [mjob],
            'Fjob': [fjob], 'reason': [reason], 'guardian': [guardian], 'traveltime': [traveltime],
            'studytime': [studytime], 'failures': [failures], 'schoolsup': ['yes' if schoolsup else 'no'],
            'famsup': ['yes' if famsup else 'no'], 'paid': ['yes' if paid else 'no'],
            'activities': ['yes' if activities else 'no'], 'nursery': ['yes' if nursery else 'no'],
            'higher': ['yes' if higher else 'no'], 'internet': ['yes' if internet else 'no'],
            'romantic': ['yes' if romantic else 'no'], 'famrel': [famrel], 'freetime': [freetime],
            'goout': [goout], 'Dalc': [dalc], 'Walc': [walc], 'health': [health], 'absences': [absences],
            'G1': [g1], 'G2': [g2]
        }, columns=features)
        
        # Preprocess input data
        input_processed, _ = preprocess_data(input_data, fit=False, le_dict=le_dict)
        
        # Make prediction
        prediction = hybrid_model.predict(input_processed)
        grade_categories = {0: "Low (0-9)", 1: "Medium (10-14)", 2: "High (15-20)"}
        predicted_grade = grade_categories[prediction[0]]
        
        # Display result
        st.subheader("Prediction Result")
        st.write(f"Predicted Performance: {predicted_grade}")
        
        # Show probability
        probs = hybrid_model.predict_proba(input_processed)[0]
        st.write("Prediction Probabilities:")
        st.write(f"- Low: {probs[0]:.2f}")
        st.write(f"- Medium: {probs[1]:.2f}")
        st.write(f"- High: {probs[2]:.2f}")
        
        # Feature importance
        st.subheader("Feature Importance (Top 10)")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': hybrid_model.rf.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        st.table(feature_importance)

if __name__ == "__main__":
    main()
