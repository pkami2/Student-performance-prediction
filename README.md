
# 🎓 Student Performance Prediction using  Hybrid Ensemble

This project uses a dynamic weighted hybrid ensemble model combining **Random Forest**, **XGBoost**, and a **Neural Network** to predict student performance. The app provides a user-friendly interface using **Streamlit** for visualizing predictions, adjusting weights, and understanding feature importance.
DEPLOYMENT LINK: https://student-performanceprediction.streamlit.app/

## 🚀 Features

- Upload your own dataset or use a sample
- Adjust ensemble weights (Random Forest, XGBoost, Neural Net)
- Real-time predictions on student scores
- Visualizations: confusion matrix, feature importance
- Dynamic accuracy based on weights

## 🧪 Models Used

1. **Random Forest**
2. **XGBoost**
3. **Neural Network (Keras Sequential model)**
4. **Dynamic Weighted Averaging** for final prediction

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd your-repo-directory

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

## 📊 Dataset

The sample dataset (`student_data.csv`) includes the following:
- Demographics: Gender, Age
- Academic Info: GPA, Study Hours
- Socioeconomic Info: Parental Education, Internet Access


## 🤝 Contributions

Feel free to fork and contribute! Open issues or submit pull requests for improvements.


