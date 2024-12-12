# FIC_GRL
# Main-Practice

**Master in Data Science - Machine Learning**  
**Authors:** Frida Ibarra and Gema Romero
Ruta: https://github.com/fridaibarra23/FIC_GRL.git
GitHub Frida Ibarra -fridaibarra23
Carpeta de Github que contiene la práctica -FIC_GRL, rama Main
Correo Electrónico: frida.ibarra@cunef.edu

---

## About the Dataset

This project aims to develop a model to **predict loan defaults** using individual customer data. The goal is to create a model that, based on information such as **income** and **employment status**, can estimate the probability of a customer defaulting on a loan.

It is essential to ensure that the model is **fair and unbiased**, avoiding any form of discrimination against any group of individuals. For this reason, special attention will be paid to variables like **gender**, which are legally protected and should not influence credit decisions.

---

## Business Case

### Objective: Predicting Payment Difficulties

The goal of this project is to **develop a robust predictive model** to determine the likelihood of loan approval for potential clients, based on diverse input variables such as:

- Income  
- Credit History  
- Employment Status  
- Debt-to-Income Ratio  

Accurate prediction of loan approvals can:

- **Optimize the loan evaluation process**  
- **Improve operational efficiency**  
- **Reduce default risks**  

A critical aspect of this model is **compliance with regulations prohibiting bias in decision-making processes**. Variables such as **race, gender, ethnicity**, and other protected attributes must not influence loan approval outcomes. The model will be designed to ensure **fairness by excluding these attributes from direct consideration** and implementing **fairness-aware machine learning techniques** to mitigate any indirect bias.

This approach ensures that the model aligns with **regulatory standards**, maintains **accuracy, transparency**, and **ethical fairness**, while promoting **equitable financial inclusion**.

---

### Business Case Objectives: Improving Credit Risk Management

- **Mitigating Credit Risk:** Proactively identify and address potential loan defaults.  
- **Streamlining Operations:** Automate and optimize the loan evaluation process to achieve cost savings.  
- **Personalized Customer Engagement:** Tailor offerings based on individual risk profiles.  
- **Customer Satisfaction:** Enhance customer experience through efficient and transparent processes.  
- **Maximizing Returns:** Optimize the loan portfolio for profitability while minimizing losses.  
- **Data-Driven Decision Making:** Empower leadership with actionable insights for strategic planning.

---

## Objectives and Work Plan

### 1. Exploratory Data Analysis (EDA)

- Understand the **distribution of variables** and identify patterns or anomalies that might affect payment behavior.
- Explore relationships between **key variables** to uncover potential predictors of loan difficulties.
- Investigate the **balance of the target variable** (payment difficulty: yes/no) to determine if **techniques to address class imbalance** are needed.

---

### 2. Model Development

- **Model Selection:** Evaluate multiple algorithms, such as:
  - Logistic Regression  
  - Decision Trees  
  - Random Forests  
  - Gradient Boosting  

  to identify the best-performing model.

- **Hyperparameter Tuning:** Optimize model parameters to achieve **better predictive performance**.
- **Cross-Validation:** Validate the model across **different subsets of data** to ensure its **robustness and reliability**.
- Implement the final model, evaluate it, and draw conclusions based on the obtained results.

---

### 3. Handling Imbalanced Data

- If customers facing **payment difficulties** are underrepresented, apply **resampling techniques** such as:
  - **Oversampling** (e.g., **SMOTE**)  
  - **Undersampling**  

  to balance the dataset.

- **Test the impact of these balancing techniques** on model performance to ensure **fair evaluation of predictions**.

---

### 4. Evaluation and Metrics

- Use metrics such as:
  - **AUC-ROC**  
  - **Precision**  
  - **Recall**  
  - **F1-score**  

  to assess the model’s effectiveness in **identifying payment difficulties**.

- Focus on **minimizing false negatives**, as incorrectly predicting that a customer would not face difficulties could result in **financial losses**.

---

### 5. Explainability

- Provide **clear insights into the internal workings of the model** using interpretability tools.
- Identify the **most and least influential variables** in the model’s decision-making process.
- Understand how the model **makes predictions in different cases** and how the features **influence these decisions**.
- Dive deeper into topics related to **model interpretability**, ensuring a comprehensive understanding of the model's behavior and decision-making.

---

### 6. Final Conclusions

Present the **key findings** from the analysis and modeling process. These conclusions will:

- Summarize insights about the **dataset** and the relationships between key **input variables**.
- Offer a comprehensive view of the **model's performance**, **predictive power**, and **accuracy**.
- Provide actionable insights into the **relationships between variables**, aiding in better decision-making and risk management.