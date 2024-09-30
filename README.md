# Credit Risk Assessment Using Machine Learning

This project aims to develop a machine learning model to predict credit risk for loan applicants. By analyzing historical loan data, the model predicts the likelihood of a customer defaulting on a loan, assisting financial institutions in making informed decisions when extending credit.

## Features
- Uses machine learning algorithms (Logistic Regression, Support Vector Machine, Random Forest) to predict credit risk.
- Preprocesses historical loan application data, including features such as customer demographics, credit history, and loan information.
- Compares multiple models and evaluates their performance to select the best model for predicting loan defaults.

## Technologies Used
- **Python**: Main programming language for the project.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For implementing machine learning models and evaluation.
- **Matplotlib & Seaborn**: For data visualization.
- **Jupyter Notebook**: For interactive data analysis and model building.

## Project Structure
```plaintext
├── data
│   └── bankloans.csv         # Input dataset
├── models
│   └── logistic_model.pkl     # Trained logistic regression model
├── notebooks
│   └── CreditRiskAssessment.ipynb  # Main notebook for training and evaluation
├── README.md                  # Project README file
└── requirements.txt           # Python dependencies

