 Password Strength Analyzer

This is a machine learning project that predicts password strength using different models like Logistic Regression, XGBoost, CNN, and BiLSTM.

It is based on password datasets like RockYou and LinkedIn and compares classical machine learning with deep learning approaches.

 Results

| Model               | Accuracy | Macro F1 | False Strength Rate |
| ------------------- | -------- | -------- | ------------------- |
| Zxcvbn Baseline     | 81.1%    | 0.729    | 18.2%               |
| Logistic Regression | 81.1%    | 0.786    | 14.7%               |
| XGBoost             | 86.3%    | 0.841    | 10.2%               |
| CNN                 | 89.1%    | 0.872    | 7.8%                |
| BiLSTM              | 92.4%    | 0.901    | 4.9%                |

 Project Structure

Password-Strength-ML/

* Data/

  * Download_Data.sh
  * Preprocess.py
  * Label_Passwords.py
* Features/

  * Extract_Features.py
* Models/

  * Train_LR.py
  * Train_XGBoost.py
  * Train_CNN.py
  * Train_BiLSTM.py
  * Evaluate.py
* Api/

  * App.py
* Webapp/

  * Src/App.jsx
* Requirements.txt
* Dockerfile



