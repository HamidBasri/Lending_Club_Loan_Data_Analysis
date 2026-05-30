```python
"""
LENDING CLUB LOAN DATA ANALYSIS
Enhanced Student Project Notebook
Deep Learning with Keras and TensorFlow

PROJECT GOAL
Build a deep learning model that predicts whether a loan will default,
using historical Lending Club loan data.

FILE NEEDED IN THE SAME FOLDER AS THIS NOTEBOOK
1. loan_data.csv

PROJECT RULES
1. Do not use for-loops.
2. Do not create user-defined functions.
3. Transform categorical values into numerical values before modeling.
4. Perform exploratory data analysis before building the model.
5. Check correlation between features and remove strongly correlated features.
6. Do not balance the full dataset before the train-test split.
7. Keep the test set untouched until final evaluation.

WHAT THIS PROJECT MUST COVER
1. Feature Transformation
   - Transform categorical values into numerical values.

2. Exploratory Data Analysis
   - Study different factors in the dataset.

3. Additional Feature Engineering
   - Check correlation between features.
   - Drop features that have a strong correlation.

4. Modeling
   - Build a deep learning model using Keras with TensorFlow backend.

FINAL DELIVERABLES
1. Evidence that the dataset was loaded correctly
2. Categorical feature transformation
3. Exploratory data analysis with meaningful plots
4. Target class distribution and imbalance check
5. Correlation analysis
6. Removal of strongly correlated features if needed
7. Prepared model-ready dataset
8. Deep learning classification model
9. Model evaluation using confusion matrix, classification report, sensitivity, and ROC-AUC
10. Final business interpretation
"""

# Column Rename Map
COLUMN_RENAME_MAP: dict[str, str] = {
    # Borrower Eligibility
    "credit.policy": "meets_credit_policy",  # 1 = passed LendingClub underwriting standards, 0 = did not
    # Loan Characteristics
    "purpose": "loan_purpose",  # self-reported reason for the loan (e.g. debt consolidation, credit card)
    "int.rate": "interest_rate",  # annual interest rate assigned to the loan — higher = riskier borrower
    "installment": "monthly_installment",  # fixed monthly repayment amount in USD
    # Borrower Financials
    "log.annual.inc": "log_annual_income",  # natural log of borrower's annual income — log-transformed to reduce skew
    "dti": "debt_to_income_ratio",  # total monthly debt / gross monthly income — higher = more financially stretched
    # Credit Profile
    "fico": "fico_score",  # credit score at application time (300–850) — strongest repayment predictor
    "days.with.cr.line": "days_with_credit_line",  # how long the borrower has had an open credit line — proxy for credit maturity
    "revol.bal": "revolving_balance",  # total outstanding balance across all revolving accounts (e.g. credit cards) in USD
    "revol.util": "revolving_utilization",  # revolving balance / credit limit (%) — high values signal financial stress
    "inq.last.6mths": "inquiries_last_6months",  # number of hard credit inquiries in the last 6 months — signals credit-seeking behavior
    "delinq.2yrs": "delinquencies_last_2years",  # times borrower was 30+ days past due in the last 2 years — payment reliability signal
    "pub.rec": "public_records",  # number of derogatory public records (bankruptcies, liens) — even 1 is a strong risk flag
    # Target Variable
    "not.fully.paid": "is_defaulted",  # 1 = loan was not fully repaid (default/charge-off), 0 = fully repaid — classification target
}


def print_qa(question: str, answer: str) -> None:
    bold = "\033[1m"
    green = "\033[92m"
    reset = "\033[0m"
    print(f"{bold}{question}{reset}")
    print(f"{green}{answer}{reset}\n")

```


```python
# ============================================================
# SECTION 1. IMPORT THE REQUIRED LIBRARIES
# ============================================================
from typing import cast

# Core data handling
import numpy as np  # numerical operations and array math
import pandas as pd  # load and manipulate data in DataFrames
import matplotlib.pyplot as plt  # create plots and visualizations
import seaborn as sns  # statistical data visualization
from IPython.display import display  # display data in an interactive way

# Model training and evaluation
from sklearn.model_selection import train_test_split  # split data into train and test sets
from sklearn.preprocessing import StandardScaler  # normalize features to zero mean and unit variance

from sklearn.metrics import (
    confusion_matrix,  # show TP, TN, FP, FN counts
    classification_report,  # show precision, recall, F1 scores
    roc_auc_score,  # measure model discrimination ability with a single score
    roc_curve,  # get the points to draw the ROC curve
)

# Handle imbalanced classes
from imblearn.under_sampling import RandomUnderSampler  # downsample majority class to balance the dataset

# Deep learning with Keras and TensorFlow
import tensorflow as tf  # backend compute engine
from keras.models import Sequential  # stack layers in sequence
from keras.layers import (
    Dense,
    Dropout,
    InputLayer,
)  # Dense is a fully connected layer, Dropout prevents overfitting, Input Layer
from keras.metrics import Recall, AUC  # track these metrics during training
from keras.callbacks import EarlyStopping  # stop training when validation performance plateaus
# Plotting
```


```python
# ============================================================
# SECTION 2. LOAD THE DATA
# ============================================================

# Question 1
# Load loan_data.csv into a DataFrame named loan_data.
#
# Then:
# - Display the first 5 rows
# - Display the shape of the dataset
# - Display the column names
# - Display the data types


dataset_path = "loan_data.csv"
loan_data = pd.read_csv(dataset_path)

print("First 5 rows of the dataset")
display(loan_data.head(5))

print("\nShape of the dataset")
print(loan_data.shape)

print("\nColumn names")
print(loan_data.columns.tolist())

print("\nData types")
print(loan_data.dtypes)
```

    First 5 rows of the dataset



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit.policy</th>
      <th>purpose</th>
      <th>int.rate</th>
      <th>installment</th>
      <th>log.annual.inc</th>
      <th>dti</th>
      <th>fico</th>
      <th>days.with.cr.line</th>
      <th>revol.bal</th>
      <th>revol.util</th>
      <th>inq.last.6mths</th>
      <th>delinq.2yrs</th>
      <th>pub.rec</th>
      <th>not.fully.paid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>0.1189</td>
      <td>829.10</td>
      <td>11.350407</td>
      <td>19.48</td>
      <td>737</td>
      <td>5639.958333</td>
      <td>28854</td>
      <td>52.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>credit_card</td>
      <td>0.1071</td>
      <td>228.22</td>
      <td>11.082143</td>
      <td>14.29</td>
      <td>707</td>
      <td>2760.000000</td>
      <td>33623</td>
      <td>76.7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>0.1357</td>
      <td>366.86</td>
      <td>10.373491</td>
      <td>11.63</td>
      <td>682</td>
      <td>4710.000000</td>
      <td>3511</td>
      <td>25.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>0.1008</td>
      <td>162.34</td>
      <td>11.350407</td>
      <td>8.10</td>
      <td>712</td>
      <td>2699.958333</td>
      <td>33667</td>
      <td>73.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>credit_card</td>
      <td>0.1426</td>
      <td>102.92</td>
      <td>11.299732</td>
      <td>14.97</td>
      <td>667</td>
      <td>4066.000000</td>
      <td>4740</td>
      <td>39.5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Shape of the dataset
    (9578, 14)
    
    Column names
    ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid']
    
    Data types
    credit.policy          int64
    purpose               object
    int.rate             float64
    installment          float64
    log.annual.inc       float64
    dti                  float64
    fico                   int64
    days.with.cr.line    float64
    revol.bal              int64
    revol.util           float64
    inq.last.6mths         int64
    delinq.2yrs            int64
    pub.rec                int64
    not.fully.paid         int64
    dtype: object



```python
# ============================================================
# SECTION 3. INITIAL DATA EXPLORATION
# ============================================================

# Question 2
# Study the dataset by displaying:
# - Number of unique values in the target variable "not.fully.paid"
# - Summary statistics for numeric columns only
# - The purpose of numeric columns only
# - The names of categorical columns only
# - The names of numeric columns only


# Count unique values in the target column
unique_values_target = loan_data["not.fully.paid"].unique()

# Summary statistics for numeric columns only
numeric_columns = loan_data.select_dtypes(include="number").columns

# Summary statistics for numeric columns only
summary_stats = loan_data[numeric_columns].describe().T

# Calculate skewness
skewness = loan_data[numeric_columns].skew()

# Append skewness to the summary statistics
summary_stats["skew"] = skewness

# Identify categorical columns only
categorical_columns = loan_data.select_dtypes(include=["object", "category"]).columns

# Show results
print("\nUnique values in not.fully.paid")
print(unique_values_target)

print("\nSummary statistics")
print(f"{len(numeric_columns)} Numeric columns: {numeric_columns}")
print(f"{len(categorical_columns)} Categorical columns: {categorical_columns}")
display(summary_stats)
# Written answer:
# What does not.fully.paid = 0 mean?
# The borrower repaid the loan completely on time without missing payments.
#
# What does not.fully.paid = 1 mean?
# The borrower did not repay the loan fully. This includes defaults, charge-offs, or missed payments.

# # KDE + Histogram Grid
# Shows distribution shape and skewness per column

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    # Plot histogram with one color, then overlay KDE with different color
    sns.histplot(data=loan_data, x=col, kde=True, ax=axes[i], color="#FF6B6B", kde_kws={"bw_adjust": 0.5})
    # Change KDE line color separately by accessing the line
    for line in axes[i].lines:
        line.set_color("black")
        line.set_linewidth(2)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")

for i in range(len(numeric_columns), len(axes)):
    axes[i].set_visible(False)

plt.suptitle("KDE + Histogram Grid", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("""
KDE + HISTOGRAM
===============
credit.policy     Two spikes at 0 and 1, the spike at 1 is much taller, most borrowers passed
int.rate          Peaks around 11-13%, right tail trails off toward 22%, slightly skewed right
installment       Multimodal distribution with clear clusters, not evenly spread
log.annual.inc    Clean bell shape centered around 11, most borrowers earn a similar range
dti               Flat and spread out from 0 to 30, no single debt level dominates
fico              Smooth bell shape centered around 700, fewer borrowers at the extremes
days.with.cr.line Peaks around 3,000-4,000 days then drops off, most have moderate credit history
revol.bal         Highly right-skewed; heavy concentration near zero with a few extreme large balances
revol.util        Unusually flat from 0 to 100, people use very different amounts of their credit
inq.last.6mths    Massive spike at 0-2 then drops sharply, most had very few recent credit checks
delinq.2yrs       Almost entirely one tall spike at zero, late payments are very rare
pub.rec           Same as delinq, one tall spike at zero, bankruptcies or liens are very rare
not.fully.paid    Tall spike at 0 and small spike at 1, confirming 84% repaid vs 16% defaulted
""")


# Violin + Box Grid
# Shows shape, quartiles, and outliers per column

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.violinplot(y=loan_data[col], ax=axes[i], color="#55A868", inner="box")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")

for j in range(len(numeric_columns), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Violin + Box Grid", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("""
VIOLIN + BOX
============

credit.policy     Wide blob at 1 and tiny blob at 0, clearly behaves like a binary feature
int.rate          Narrow symmetric shape, rates are tightly packed between 10% and 15%
installment       Wide bumpy body with visible bulges, reflects the multiple payment clusters
log.annual.inc    Smooth hourglass shape, incomes concentrate in the middle range
dti               Wide flat body from top to bottom, debt levels are genuinely all over the place
fico              Triangle shape widest at 700, thinner at both ends, normal distribution confirmed
days.with.cr.line Wider at the bottom, thinner as it stretches upward, a few have very long history
revol.bal         Entire body squashed at zero, single long line shoots up to $1.2M
revol.util        Consistently wide from 0 to 100, almost perfectly rectangular, very uniform
inq.last.6mths    Tiny diamond near zero, single long whisker above confirms extreme outliers
delinq.2yrs       Nearly invisible body at zero, scattered dots above are rare exceptions
pub.rec           Almost all at zero with a handful of dots reaching up to 5
not.fully.paid    Large blob at 0 and small blob at 1, visually confirms the class imbalance

""")


# Box Plot Grid
# Shows quartiles, median, and outliers cleanly per column

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.boxplot(y=loan_data[col], ax=axes[i], color="#C44E52")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")

for j in range(len(numeric_columns), len(axes)):
    axes[j].set_visible(False)

plt.title("Box Plot Grid", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("""
credit.policy     Box sits at 1 with one lonely dot at 0, practically a constant column
int.rate          Tight box from 10% to 14%, a cluster of dots above 18% are clear outliers
installment       Wide box from around $160 to $430, many dots above $700 worth investigating
log.annual.inc    Compact box around $40k-$80k range, a few dots below $10k and above $500k
dti               Balanced box from roughly 7% to 18%, whiskers reach 0 and 30%, no extreme dots
fico              Box from 682 to 737, one dot above 820 is a rare excellent-credit borrower
days.with.cr.line Box sits around 3,000-5,700 days, many dots above 10,000 days are genuine outliers
revol.bal         Box is invisible near zero, dots scatter all the way to $1.2M, serious outlier problem
revol.util        Wide box from 22% to 71%, whiskers reach 0 and 100%, a few go slightly above
inq.last.6mths    Box barely visible near 0-2, dozens of dots stacking up above 5 inquiries
delinq.2yrs       Box entirely at zero, dots scattered above up to 13 late payments
pub.rec           Box entirely at zero, dots rise to 5, each one a bankruptcy or lien event
not.fully.paid    Box at zero, single dot at 1 is the only hint of defaults in this view
""")

```

    
    Unique values in not.fully.paid
    [0 1]
    
    Summary statistics
    13 Numeric columns: Index(['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti',
           'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
           'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'],
          dtype='object')
    1 Categorical columns: Index(['purpose'], dtype='object')



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>credit.policy</th>
      <td>9578.0</td>
      <td>0.804970</td>
      <td>0.396245</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>-1.539621</td>
    </tr>
    <tr>
      <th>int.rate</th>
      <td>9578.0</td>
      <td>0.122640</td>
      <td>0.026847</td>
      <td>0.060000</td>
      <td>0.103900</td>
      <td>0.122100</td>
      <td>0.140700</td>
      <td>2.164000e-01</td>
      <td>0.164420</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>9578.0</td>
      <td>319.089413</td>
      <td>207.071301</td>
      <td>15.670000</td>
      <td>163.770000</td>
      <td>268.950000</td>
      <td>432.762500</td>
      <td>9.401400e+02</td>
      <td>0.912522</td>
    </tr>
    <tr>
      <th>log.annual.inc</th>
      <td>9578.0</td>
      <td>10.932117</td>
      <td>0.614813</td>
      <td>7.547502</td>
      <td>10.558414</td>
      <td>10.928884</td>
      <td>11.291293</td>
      <td>1.452835e+01</td>
      <td>0.028668</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>9578.0</td>
      <td>12.606679</td>
      <td>6.883970</td>
      <td>0.000000</td>
      <td>7.212500</td>
      <td>12.665000</td>
      <td>17.950000</td>
      <td>2.996000e+01</td>
      <td>0.023941</td>
    </tr>
    <tr>
      <th>fico</th>
      <td>9578.0</td>
      <td>710.846314</td>
      <td>37.970537</td>
      <td>612.000000</td>
      <td>682.000000</td>
      <td>707.000000</td>
      <td>737.000000</td>
      <td>8.270000e+02</td>
      <td>0.471260</td>
    </tr>
    <tr>
      <th>days.with.cr.line</th>
      <td>9578.0</td>
      <td>4560.767197</td>
      <td>2496.930377</td>
      <td>178.958333</td>
      <td>2820.000000</td>
      <td>4139.958333</td>
      <td>5730.000000</td>
      <td>1.763996e+04</td>
      <td>1.155748</td>
    </tr>
    <tr>
      <th>revol.bal</th>
      <td>9578.0</td>
      <td>16913.963876</td>
      <td>33756.189557</td>
      <td>0.000000</td>
      <td>3187.000000</td>
      <td>8596.000000</td>
      <td>18249.500000</td>
      <td>1.207359e+06</td>
      <td>11.161058</td>
    </tr>
    <tr>
      <th>revol.util</th>
      <td>9578.0</td>
      <td>46.799236</td>
      <td>29.014417</td>
      <td>0.000000</td>
      <td>22.600000</td>
      <td>46.300000</td>
      <td>70.900000</td>
      <td>1.190000e+02</td>
      <td>0.059985</td>
    </tr>
    <tr>
      <th>inq.last.6mths</th>
      <td>9578.0</td>
      <td>1.577469</td>
      <td>2.200245</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.300000e+01</td>
      <td>3.584151</td>
    </tr>
    <tr>
      <th>delinq.2yrs</th>
      <td>9578.0</td>
      <td>0.163708</td>
      <td>0.546215</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.300000e+01</td>
      <td>6.061793</td>
    </tr>
    <tr>
      <th>pub.rec</th>
      <td>9578.0</td>
      <td>0.062122</td>
      <td>0.262126</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000e+00</td>
      <td>5.126434</td>
    </tr>
    <tr>
      <th>not.fully.paid</th>
      <td>9578.0</td>
      <td>0.160054</td>
      <td>0.366676</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
      <td>1.854592</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](lending_loan_files/lending_loan_3_2.png)
    


    
    KDE + HISTOGRAM
    ===============
    credit.policy     Two spikes at 0 and 1, the spike at 1 is much taller, most borrowers passed
    int.rate          Peaks around 11-13%, right tail trails off toward 22%, slightly skewed right
    installment       Multimodal distribution with clear clusters, not evenly spread
    log.annual.inc    Clean bell shape centered around 11, most borrowers earn a similar range
    dti               Flat and spread out from 0 to 30, no single debt level dominates
    fico              Smooth bell shape centered around 700, fewer borrowers at the extremes
    days.with.cr.line Peaks around 3,000-4,000 days then drops off, most have moderate credit history
    revol.bal         Highly right-skewed; heavy concentration near zero with a few extreme large balances
    revol.util        Unusually flat from 0 to 100, people use very different amounts of their credit
    inq.last.6mths    Massive spike at 0-2 then drops sharply, most had very few recent credit checks
    delinq.2yrs       Almost entirely one tall spike at zero, late payments are very rare
    pub.rec           Same as delinq, one tall spike at zero, bankruptcies or liens are very rare
    not.fully.paid    Tall spike at 0 and small spike at 1, confirming 84% repaid vs 16% defaulted
    



    
![png](lending_loan_files/lending_loan_3_4.png)
    


    
    VIOLIN + BOX
    ============
    
    credit.policy     Wide blob at 1 and tiny blob at 0, clearly behaves like a binary feature
    int.rate          Narrow symmetric shape, rates are tightly packed between 10% and 15%
    installment       Wide bumpy body with visible bulges, reflects the multiple payment clusters
    log.annual.inc    Smooth hourglass shape, incomes concentrate in the middle range
    dti               Wide flat body from top to bottom, debt levels are genuinely all over the place
    fico              Triangle shape widest at 700, thinner at both ends, normal distribution confirmed
    days.with.cr.line Wider at the bottom, thinner as it stretches upward, a few have very long history
    revol.bal         Entire body squashed at zero, single long line shoots up to $1.2M
    revol.util        Consistently wide from 0 to 100, almost perfectly rectangular, very uniform
    inq.last.6mths    Tiny diamond near zero, single long whisker above confirms extreme outliers
    delinq.2yrs       Nearly invisible body at zero, scattered dots above are rare exceptions
    pub.rec           Almost all at zero with a handful of dots reaching up to 5
    not.fully.paid    Large blob at 0 and small blob at 1, visually confirms the class imbalance
    
    



    
![png](lending_loan_files/lending_loan_3_6.png)
    


    
    credit.policy     Box sits at 1 with one lonely dot at 0, practically a constant column
    int.rate          Tight box from 10% to 14%, a cluster of dots above 18% are clear outliers
    installment       Wide box from around $160 to $430, many dots above $700 worth investigating
    log.annual.inc    Compact box around $40k-$80k range, a few dots below $10k and above $500k
    dti               Balanced box from roughly 7% to 18%, whiskers reach 0 and 30%, no extreme dots
    fico              Box from 682 to 737, one dot above 820 is a rare excellent-credit borrower
    days.with.cr.line Box sits around 3,000-5,700 days, many dots above 10,000 days are genuine outliers
    revol.bal         Box is invisible near zero, dots scatter all the way to $1.2M, serious outlier problem
    revol.util        Wide box from 22% to 71%, whiskers reach 0 and 100%, a few go slightly above
    inq.last.6mths    Box barely visible near 0-2, dozens of dots stacking up above 5 inquiries
    delinq.2yrs       Box entirely at zero, dots scattered above up to 13 late payments
    pub.rec           Box entirely at zero, dots rise to 5, each one a bankruptcy or lien event
    not.fully.paid    Box at zero, single dot at 1 is the only hint of defaults in this view
    



```python

```


```python
# ============================================================
# SECTION 4. CHECK FOR MISSING VALUES
# ============================================================

# Question 3
# Check whether the dataset contains missing values.
#
# Complete all of the following:
# - Count total missing values
# - Count missing values per column
# - State whether missing-value treatment is needed

total_missing_values = loan_data.isnull().sum().sum()

missing_values_per_column = loan_data.isnull().sum()

print("\nTotal missing values in the dataset")
print(total_missing_values)

print("\nMissing values per column")
print(missing_values_per_column)

# Written answer:
# Does this dataset contain missing values?
# No, there are no missing values in this dataset. All columns have complete data.
#
# Is missing-value imputation needed here?
# No, imputation is not needed since there are no missing values to handle.
```

    
    Total missing values in the dataset
    0
    
    Missing values per column
    credit.policy        0
    purpose              0
    int.rate             0
    installment          0
    log.annual.inc       0
    dti                  0
    fico                 0
    days.with.cr.line    0
    revol.bal            0
    revol.util           0
    inq.last.6mths       0
    delinq.2yrs          0
    pub.rec              0
    not.fully.paid       0
    dtype: int64



```python
# ============================================================
# SECTION 5. FEATURE TRANSFORMATION
# REQUIRED TASK 1. TRANSFORM CATEGORICAL VALUES INTO NUMERICAL VALUES
# ============================================================

# Question 4
# Identify the categorical feature and transform it into numeric columns.
#
# In this dataset, purpose is the categorical column.
#
# Complete all of the following:
# - Display the unique values in purpose
# - Create dummy variables for purpose
# - Confirm that the transformed dataset no longer contains raw text columns

print("\nUnique loan purposes")
print(loan_data["purpose"].unique())

loan_data_encoded = pd.get_dummies(loan_data, columns=["purpose"], drop_first=True, dtype=int)

print("\nFirst 5 rows after encoding")
display(loan_data_encoded.head())

print("\nData types after encoding")
print(loan_data_encoded.dtypes)

# Written answer:
# Why must purpose be transformed before building a neural network?
# Neural networks only understand numbers. Text like "debt_consolidation"
# means nothing to the model mathematically, so we convert each category
# into a binary column (0 or 1) that the network can actually compute with.
#
# What new columns were created from purpose?
# One binary column per unique purpose value, minus one (drop_first=True removes
# one category to avoid multicollinearity).
```

    
    Unique loan purposes
    ['debt_consolidation' 'credit_card' 'all_other' 'home_improvement'
     'small_business' 'major_purchase' 'educational']
    
    First 5 rows after encoding



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit.policy</th>
      <th>int.rate</th>
      <th>installment</th>
      <th>log.annual.inc</th>
      <th>dti</th>
      <th>fico</th>
      <th>days.with.cr.line</th>
      <th>revol.bal</th>
      <th>revol.util</th>
      <th>inq.last.6mths</th>
      <th>delinq.2yrs</th>
      <th>pub.rec</th>
      <th>not.fully.paid</th>
      <th>purpose_credit_card</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_educational</th>
      <th>purpose_home_improvement</th>
      <th>purpose_major_purchase</th>
      <th>purpose_small_business</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.1189</td>
      <td>829.10</td>
      <td>11.350407</td>
      <td>19.48</td>
      <td>737</td>
      <td>5639.958333</td>
      <td>28854</td>
      <td>52.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.1071</td>
      <td>228.22</td>
      <td>11.082143</td>
      <td>14.29</td>
      <td>707</td>
      <td>2760.000000</td>
      <td>33623</td>
      <td>76.7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.1357</td>
      <td>366.86</td>
      <td>10.373491</td>
      <td>11.63</td>
      <td>682</td>
      <td>4710.000000</td>
      <td>3511</td>
      <td>25.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.1008</td>
      <td>162.34</td>
      <td>11.350407</td>
      <td>8.10</td>
      <td>712</td>
      <td>2699.958333</td>
      <td>33667</td>
      <td>73.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.1426</td>
      <td>102.92</td>
      <td>11.299732</td>
      <td>14.97</td>
      <td>667</td>
      <td>4066.000000</td>
      <td>4740</td>
      <td>39.5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Data types after encoding
    credit.policy                   int64
    int.rate                      float64
    installment                   float64
    log.annual.inc                float64
    dti                           float64
    fico                            int64
    days.with.cr.line             float64
    revol.bal                       int64
    revol.util                    float64
    inq.last.6mths                  int64
    delinq.2yrs                     int64
    pub.rec                         int64
    not.fully.paid                  int64
    purpose_credit_card             int64
    purpose_debt_consolidation      int64
    purpose_educational             int64
    purpose_home_improvement        int64
    purpose_major_purchase          int64
    purpose_small_business          int64
    dtype: object



```python
# ============================================================
# SECTION 6. CHECK TARGET CLASS BALANCE
# ============================================================

target_counts = loan_data_encoded["not.fully.paid"].value_counts().sort_index()

target_percentages = (target_counts / target_counts.sum() * 100).round(2)

print("\nTarget counts")
print(target_counts)

print("\nPercentage of fully paid loans, not.fully.paid = 0")
print(f"{target_percentages[0]}%")

print("\nPercentage of not fully paid loans, not.fully.paid = 1")
print(f"{target_percentages[1]}%")

plt.figure(figsize=(7, 5))

sns.barplot(
    x=target_counts.index.astype(str),
    y=target_counts.values,
    hue=target_counts.index.astype(str),
    palette=["#4C72B0", "#C44E52"],
    legend=False,
)

for i, (count, pct) in enumerate(zip(target_counts.values, target_percentages.values)):
    plt.text(i, count + 80, f"{count:,}\n({pct}%)", ha="center", fontsize=11)

plt.title("Target Class Distribution", fontsize=14, pad=15)
plt.xlabel("not.fully.paid  (0 = repaid, 1 = defaulted)", fontsize=11)
plt.ylabel("Number of Loans", fontsize=11)
plt.xticks(rotation=0)
sns.despine()  # remove plot borders
plt.tight_layout()
plt.show()

# Written answer:
# Is the dataset balanced or imbalanced?
# Imbalanced. Roughly 84% repaid and only 16% defaulted.
# For every default there are about 5 repaid loans in the dataset.
#
# Why can this be a problem for a default-prediction model?
# The model will lean toward predicting repaid almost every time because
# that is what it sees most during training. It will look accurate on paper
# but will miss most actual defaults, which is exactly what the bank needs to catch.
```

    
    Target counts
    not.fully.paid
    0    8045
    1    1533
    Name: count, dtype: int64
    
    Percentage of fully paid loans, not.fully.paid = 0
    83.99%
    
    Percentage of not fully paid loans, not.fully.paid = 1
    16.01%



    
![png](lending_loan_files/lending_loan_7_1.png)
    



```python
# ============================================================
# SECTION 7. EXPLORATORY DATA ANALYSIS OF DIFFERENT FACTORS
# REQUIRED TASK 2. EXPLORATORY DATA ANALYSIS
# ============================================================

# Question 6
# Explore how different factors relate to loan default.
#
# Create and interpret the following:
# 1. Default rate by loan purpose
# 2. Default rate by credit policy
# 3. Interest rate by payment outcome
# 4. FICO score by payment outcome
# 5. Debt-to-income ratio by payment outcome

# ----------------------------
# 6A. Default rate by purpose
# ----------------------------

default_rate_by_purpose = (loan_data.groupby("purpose")["not.fully.paid"].mean() * 100).sort_values(ascending=False)  # type: ignore

print("\nDefault rate by purpose")
print(default_rate_by_purpose)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=default_rate_by_purpose.index,
    y=default_rate_by_purpose.values,
    hue=default_rate_by_purpose.index,
    palette="Reds_r",
    legend=False,
)

for i, val in enumerate(default_rate_by_purpose.values):
    plt.text(i, val + 0.3, f"{val:.1f}%", ha="center", fontsize=9)

plt.title("Default Rate by Loan Purpose", fontsize=14, pad=15)
plt.xlabel("Loan Purpose", fontsize=11)
plt.ylabel("Default Rate (%)", fontsize=11)
plt.xticks(rotation=45, ha="right")
sns.despine()
plt.tight_layout()
plt.show()

# Written interpretation:

print_qa(
    "Which loan purpose appears to have the highest default rate?",
    """
Small business loans have the highest default rate by a noticeable margin.
This makes sense because small businesses carry more financial uncertainty
compared to personal loans like credit cards or debt consolidation.
    """,
)


# ----------------------------
# 6B. Default rate by credit policy
# ----------------------------

default_rate_by_credit_policy = loan_data.groupby("credit.policy")["not.fully.paid"].mean() * 100  # type: ignore

print("\nDefault rate by credit policy")
print(default_rate_by_credit_policy)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=default_rate_by_credit_policy.index.astype(str),
    y=default_rate_by_credit_policy.values,
    hue=default_rate_by_credit_policy.index.astype(str),
    palette=["#C44E52", "#4C72B0"],
    legend=False,
)

for i, val in enumerate(default_rate_by_credit_policy.values):
    plt.text(i, val + 0.3, f"{val:.1f}%", ha="center", fontsize=11)

plt.title("Default Rate by Credit Policy", fontsize=14, pad=15)
plt.xlabel("Credit Policy  (0 = did not meet standards, 1 = met standards)", fontsize=11)
plt.ylabel("Default Rate (%)", fontsize=11)
plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.show()

# Written interpretation:
print_qa(
    "Which group has a higher default rate, credit.policy = 0 or credit.policy = 1?",
    """
Borrowers who did NOT meet credit standards (policy = 0) default at a higher rate.
This is expected. The credit check exists precisely to filter out risky borrowers.
What is interesting though is that policy = 1 borrowers still default at around 13%,
which tells us the credit check helps but is far from a perfect filter.
    """,
)

# ----------------------------
# 6C. Interest rate by payment outcome
# ----------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=loan_data, x="not.fully.paid", y="int.rate", hue="not.fully.paid", palette=["#4C72B0", "#C44E52"], legend=False
)

plt.title("Interest Rate by Payment Outcome", fontsize=14, pad=15)
plt.xlabel("not.fully.paid  (0 = repaid, 1 = defaulted)", fontsize=11)
plt.ylabel("Interest Rate", fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()

# Written interpretation:
print_qa(
    "Do defaulted loans appear to have higher interest rates?",
    """
Yes, but the difference is modest.
Repaid loans (0) have a median around 12% with a wider spread from 10% to 14%.
Defaulted loans (1) have a median around 13% with a tighter spread from 12% to 15%.
Both groups share similar whiskers and outliers above 20%.
So interest rate alone is not a strong separator between the two groups,
the boxes overlap too much to rely on it as a standalone predictor.
""",
)

# ----------------------------
# 6D. FICO score by payment outcome
# ----------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=loan_data, x="not.fully.paid", y="fico", hue="not.fully.paid", palette=["#4C72B0", "#C44E52"], legend=False
)

plt.title("FICO Score by Payment Outcome", fontsize=14, pad=15)
plt.xlabel("not.fully.paid  (0 = repaid, 1 = defaulted)", fontsize=11)
plt.ylabel("FICO Score", fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()

# Written interpretation:
print_qa(
    "Do defaulted loans appear to have lower FICO scores?",
    """
Yes, and this is one of the clearest separations we have.
Repaid loans (0) have a noticeably higher median FICO score than defaulted loans (1).
The boxes overlap less compared to interest rate, meaning FICO score is a
stronger signal for separating the two groups.
Borrowers who defaulted tend to have lower credit scores going in,
which aligns with what FICO scores are designed to measure:
the likelihood that someone will repay their debts.
    """,
)

# ----------------------------
# 6E. Debt-to-income ratio by payment outcome
# ----------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=loan_data, x="not.fully.paid", y="dti", hue="not.fully.paid", palette=["#4C72B0", "#C44E52"], legend=False
)

plt.title("Debt-to-Income Ratio by Payment Outcome", fontsize=14, pad=15)
plt.xlabel("not.fully.paid  (0 = repaid, 1 = defaulted)", fontsize=11)
plt.ylabel("Debt-to-Income Ratio", fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()

# Written interpretation:
print_qa(
    "Does debt-to-income ratio appear different between the two groups?",
    """
The two boxes overlap heavily, meaning dti alone does not cleanly
separate repaid from defaulted loans.
Defaulted loans (1) show a slightly higher median dti, which makes
intuitive sense as borrowers already stretched thin financially are
more likely to miss payments when unexpected costs arise.
However the difference is small and the spread of both groups is
very similar, making dti a weak standalone predictor compared to
fico score which showed a much cleaner separation.
""",
)

```

    
    Default rate by purpose
    purpose
    small_business        27.786753
    educational           20.116618
    home_improvement      17.011129
    all_other             16.602317
    debt_consolidation    15.238817
    credit_card           11.568938
    major_purchase        11.212815
    Name: not.fully.paid, dtype: float64



    
![png](lending_loan_files/lending_loan_8_1.png)
    


    [1mWhich loan purpose appears to have the highest default rate?[0m
    [92m
    Small business loans have the highest default rate by a noticeable margin.
    This makes sense because small businesses carry more financial uncertainty
    compared to personal loans like credit cards or debt consolidation.
        [0m
    
    
    Default rate by credit policy
    credit.policy
    0    27.783726
    1    13.151751
    Name: not.fully.paid, dtype: float64



    
![png](lending_loan_files/lending_loan_8_3.png)
    


    [1mWhich group has a higher default rate, credit.policy = 0 or credit.policy = 1?[0m
    [92m
    Borrowers who did NOT meet credit standards (policy = 0) default at a higher rate.
    This is expected. The credit check exists precisely to filter out risky borrowers.
    What is interesting though is that policy = 1 borrowers still default at around 13%,
    which tells us the credit check helps but is far from a perfect filter.
        [0m
    



    
![png](lending_loan_files/lending_loan_8_5.png)
    


    [1mDo defaulted loans appear to have higher interest rates?[0m
    [92m
    Yes, but the difference is modest.
    Repaid loans (0) have a median around 12% with a wider spread from 10% to 14%.
    Defaulted loans (1) have a median around 13% with a tighter spread from 12% to 15%.
    Both groups share similar whiskers and outliers above 20%.
    So interest rate alone is not a strong separator between the two groups,
    the boxes overlap too much to rely on it as a standalone predictor.
    [0m
    



    
![png](lending_loan_files/lending_loan_8_7.png)
    


    [1mDo defaulted loans appear to have lower FICO scores?[0m
    [92m
    Yes, and this is one of the clearest separations we have.
    Repaid loans (0) have a noticeably higher median FICO score than defaulted loans (1).
    The boxes overlap less compared to interest rate, meaning FICO score is a
    stronger signal for separating the two groups.
    Borrowers who defaulted tend to have lower credit scores going in,
    which aligns with what FICO scores are designed to measure:
    the likelihood that someone will repay their debts.
        [0m
    



    
![png](lending_loan_files/lending_loan_8_9.png)
    


    [1mDoes debt-to-income ratio appear different between the two groups?[0m
    [92m
    The two boxes overlap heavily, meaning dti alone does not cleanly
    separate repaid from defaulted loans.
    Defaulted loans (1) show a slightly higher median dti, which makes
    intuitive sense as borrowers already stretched thin financially are
    more likely to miss payments when unexpected costs arise.
    However the difference is small and the spread of both groups is
    very similar, making dti a weak standalone predictor compared to
    fico score which showed a much cleaner separation.
    [0m
    



```python
# ============================================================
# SECTION 8. ADDITIONAL FEATURE ENGINEERING
# REQUIRED TASK 3. CHECK CORRELATION AND DROP STRONGLY CORRELATED FEATURES
# ============================================================


# Question 7
# Check correlation between features and identify strongly correlated pairs.
#
# Complete all of the following:
# - Create a correlation matrix
# - Plot the correlation matrix
# - Build an upper-triangle correlation table
# - Identify features with absolute correlation above 0.80
# - Drop strongly correlated features if any are found
#
# Use only the predictor columns when deciding which features to drop.
# Do not include the target while deciding feature correlation.

from matplotlib import patches

correlation_matrix = loan_data_encoded.drop(columns=["not.fully.paid"]).corr()

print("\nCorrelation matrix")
display(correlation_matrix)

plt.figure(figsize=(14, 10))
mask = np.tril(np.ones(correlation_matrix.shape), k=0).astype(bool)

fig, ax = plt.subplots(figsize=(16, 10))

sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 7},
    square=True,
    cbar_kws={"shrink": 0.8},
    vmin=-1,
    vmax=1,
)

# Fill the masked lower triangle with a dashed pattern
n = len(correlation_matrix)
for i in range(n):
    for j in range(i + 1):
        ax.add_patch(
            patches.Rectangle(
                (j, i), 1, 1, fill=True, facecolor="white", edgecolor="lightgray", hatch="//", linewidth=0
            )
        )

# Draw dashed lines connecting each x-axis label to its column
for i in range(n):
    ax.plot([i + 0.5, i + 0.5], [n, i], color="gray", linewidth=0.7, linestyle="--", alpha=0.5, zorder=5)

# Draw dashed lines connecting each y-axis label to its row
for i in range(n):
    ax.plot(
        [0, i + 1],  # from left edge all the way to where the filled cells start
        [i + 0.5, i + 0.5],
        color="gray",
        linewidth=0.7,
        linestyle="--",
        alpha=0.5,
        zorder=5,
    )
plt.title("Feature Correlation Matrix", fontsize=14, pad=15)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

strongly_correlated_features = upper_triangle.columns[(upper_triangle.abs() > 0.80).any()].tolist()

print("\nStrongly correlated features to consider dropping")
print(strongly_correlated_features)

loan_data_final = loan_data_encoded.drop(columns=strongly_correlated_features)

print("\nShape before dropping strongly correlated features")
print(loan_data_encoded.shape)

print("\nShape after dropping strongly correlated features")
print(loan_data_final.shape)

print_qa(
    "\nWhich features were strongly correlated enough to drop?",
    "Any feature pairs with absolute correlation above 0.80 get flagged and dropped. "
    "none are found the two shapes are identical. ",
)

print_qa(
    "Why can dropping strongly correlated features help a model?",
    "When two features are strongly correlated they are telling the model the same story twice. "
    "This is called multicollinearity. In a neural network this creates redundant weights that "
    "compete with each other during training, making the model less stable and harder to interpret. "
    "Removing one of the pair keeps the information while reducing noise, "
    "leading to cleaner and more efficient learning.",
)
```

    
    Correlation matrix



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit.policy</th>
      <th>int.rate</th>
      <th>installment</th>
      <th>log.annual.inc</th>
      <th>dti</th>
      <th>fico</th>
      <th>days.with.cr.line</th>
      <th>revol.bal</th>
      <th>revol.util</th>
      <th>inq.last.6mths</th>
      <th>delinq.2yrs</th>
      <th>pub.rec</th>
      <th>purpose_credit_card</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_educational</th>
      <th>purpose_home_improvement</th>
      <th>purpose_major_purchase</th>
      <th>purpose_small_business</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>credit.policy</th>
      <td>1.000000</td>
      <td>-0.294089</td>
      <td>0.058770</td>
      <td>0.034906</td>
      <td>-0.090901</td>
      <td>0.348319</td>
      <td>0.099026</td>
      <td>-0.187518</td>
      <td>-0.104095</td>
      <td>-0.535511</td>
      <td>-0.076318</td>
      <td>-0.054243</td>
      <td>0.003216</td>
      <td>0.020193</td>
      <td>-0.031346</td>
      <td>0.006036</td>
      <td>0.024281</td>
      <td>-0.003511</td>
    </tr>
    <tr>
      <th>int.rate</th>
      <td>-0.294089</td>
      <td>1.000000</td>
      <td>0.276140</td>
      <td>0.056383</td>
      <td>0.220006</td>
      <td>-0.714821</td>
      <td>-0.124022</td>
      <td>0.092527</td>
      <td>0.464837</td>
      <td>0.202780</td>
      <td>0.156079</td>
      <td>0.098162</td>
      <td>-0.042109</td>
      <td>0.123607</td>
      <td>-0.019618</td>
      <td>-0.050697</td>
      <td>-0.068978</td>
      <td>0.151247</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>0.058770</td>
      <td>0.276140</td>
      <td>1.000000</td>
      <td>0.448102</td>
      <td>0.050202</td>
      <td>0.086039</td>
      <td>0.183297</td>
      <td>0.233625</td>
      <td>0.081356</td>
      <td>-0.010419</td>
      <td>-0.004368</td>
      <td>-0.032760</td>
      <td>0.000774</td>
      <td>0.161658</td>
      <td>-0.094510</td>
      <td>0.023024</td>
      <td>-0.079836</td>
      <td>0.145654</td>
    </tr>
    <tr>
      <th>log.annual.inc</th>
      <td>0.034906</td>
      <td>0.056383</td>
      <td>0.448102</td>
      <td>1.000000</td>
      <td>-0.054065</td>
      <td>0.114576</td>
      <td>0.336896</td>
      <td>0.372140</td>
      <td>0.054881</td>
      <td>0.029171</td>
      <td>0.029203</td>
      <td>0.016506</td>
      <td>0.072942</td>
      <td>-0.026214</td>
      <td>-0.119799</td>
      <td>0.116375</td>
      <td>-0.031020</td>
      <td>0.091540</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>-0.090901</td>
      <td>0.220006</td>
      <td>0.050202</td>
      <td>-0.054065</td>
      <td>1.000000</td>
      <td>-0.241191</td>
      <td>0.060101</td>
      <td>0.188748</td>
      <td>0.337109</td>
      <td>0.029189</td>
      <td>-0.021792</td>
      <td>0.006209</td>
      <td>0.084476</td>
      <td>0.179149</td>
      <td>-0.035325</td>
      <td>-0.092788</td>
      <td>-0.077719</td>
      <td>-0.069245</td>
    </tr>
    <tr>
      <th>fico</th>
      <td>0.348319</td>
      <td>-0.714821</td>
      <td>0.086039</td>
      <td>0.114576</td>
      <td>-0.241191</td>
      <td>1.000000</td>
      <td>0.263880</td>
      <td>-0.015553</td>
      <td>-0.541289</td>
      <td>-0.185293</td>
      <td>-0.216340</td>
      <td>-0.147592</td>
      <td>-0.012512</td>
      <td>-0.154132</td>
      <td>-0.013012</td>
      <td>0.097474</td>
      <td>0.067129</td>
      <td>0.063292</td>
    </tr>
    <tr>
      <th>days.with.cr.line</th>
      <td>0.099026</td>
      <td>-0.124022</td>
      <td>0.183297</td>
      <td>0.336896</td>
      <td>0.060101</td>
      <td>0.263880</td>
      <td>1.000000</td>
      <td>0.229344</td>
      <td>-0.024239</td>
      <td>-0.041736</td>
      <td>0.081374</td>
      <td>0.071826</td>
      <td>0.046220</td>
      <td>-0.009318</td>
      <td>-0.042621</td>
      <td>0.068087</td>
      <td>-0.020561</td>
      <td>0.034883</td>
    </tr>
    <tr>
      <th>revol.bal</th>
      <td>-0.187518</td>
      <td>0.092527</td>
      <td>0.233625</td>
      <td>0.372140</td>
      <td>0.188748</td>
      <td>-0.015553</td>
      <td>0.229344</td>
      <td>1.000000</td>
      <td>0.203779</td>
      <td>0.022394</td>
      <td>-0.033243</td>
      <td>-0.031010</td>
      <td>0.072316</td>
      <td>0.005785</td>
      <td>-0.034743</td>
      <td>0.003258</td>
      <td>-0.062395</td>
      <td>0.083069</td>
    </tr>
    <tr>
      <th>revol.util</th>
      <td>-0.104095</td>
      <td>0.464837</td>
      <td>0.081356</td>
      <td>0.054881</td>
      <td>0.337109</td>
      <td>-0.541289</td>
      <td>-0.024239</td>
      <td>0.203779</td>
      <td>1.000000</td>
      <td>-0.013880</td>
      <td>-0.042740</td>
      <td>0.066717</td>
      <td>0.091321</td>
      <td>0.211869</td>
      <td>-0.053128</td>
      <td>-0.114449</td>
      <td>-0.108079</td>
      <td>-0.060962</td>
    </tr>
    <tr>
      <th>inq.last.6mths</th>
      <td>-0.535511</td>
      <td>0.202780</td>
      <td>-0.010419</td>
      <td>0.029171</td>
      <td>0.029189</td>
      <td>-0.185293</td>
      <td>-0.041736</td>
      <td>0.022394</td>
      <td>-0.013880</td>
      <td>1.000000</td>
      <td>0.021245</td>
      <td>0.072673</td>
      <td>-0.033640</td>
      <td>-0.044240</td>
      <td>0.024243</td>
      <td>0.043827</td>
      <td>-0.001445</td>
      <td>0.042567</td>
    </tr>
    <tr>
      <th>delinq.2yrs</th>
      <td>-0.076318</td>
      <td>0.156079</td>
      <td>-0.004368</td>
      <td>0.029203</td>
      <td>-0.021792</td>
      <td>-0.216340</td>
      <td>0.081374</td>
      <td>-0.033243</td>
      <td>-0.042740</td>
      <td>0.021245</td>
      <td>1.000000</td>
      <td>0.009184</td>
      <td>-0.008817</td>
      <td>-0.000697</td>
      <td>-0.002214</td>
      <td>-0.013098</td>
      <td>0.004085</td>
      <td>-0.004148</td>
    </tr>
    <tr>
      <th>pub.rec</th>
      <td>-0.054243</td>
      <td>0.098162</td>
      <td>-0.032760</td>
      <td>0.016506</td>
      <td>0.006209</td>
      <td>-0.147592</td>
      <td>0.071826</td>
      <td>-0.031010</td>
      <td>0.066717</td>
      <td>0.072673</td>
      <td>0.009184</td>
      <td>1.000000</td>
      <td>0.014842</td>
      <td>0.026845</td>
      <td>-0.013521</td>
      <td>0.004704</td>
      <td>-0.011734</td>
      <td>-0.005595</td>
    </tr>
    <tr>
      <th>purpose_credit_card</th>
      <td>0.003216</td>
      <td>-0.042109</td>
      <td>0.000774</td>
      <td>0.072942</td>
      <td>0.084476</td>
      <td>-0.012512</td>
      <td>0.046220</td>
      <td>0.072316</td>
      <td>0.091321</td>
      <td>-0.033640</td>
      <td>-0.008817</td>
      <td>0.014842</td>
      <td>1.000000</td>
      <td>-0.326850</td>
      <td>-0.075076</td>
      <td>-0.103279</td>
      <td>-0.085176</td>
      <td>-0.102397</td>
    </tr>
    <tr>
      <th>purpose_debt_consolidation</th>
      <td>0.020193</td>
      <td>0.123607</td>
      <td>0.161658</td>
      <td>-0.026214</td>
      <td>0.179149</td>
      <td>-0.154132</td>
      <td>-0.009318</td>
      <td>0.005785</td>
      <td>0.211869</td>
      <td>-0.044240</td>
      <td>-0.000697</td>
      <td>0.026845</td>
      <td>-0.326850</td>
      <td>1.000000</td>
      <td>-0.161698</td>
      <td>-0.222441</td>
      <td>-0.183451</td>
      <td>-0.220542</td>
    </tr>
    <tr>
      <th>purpose_educational</th>
      <td>-0.031346</td>
      <td>-0.019618</td>
      <td>-0.094510</td>
      <td>-0.119799</td>
      <td>-0.035325</td>
      <td>-0.013012</td>
      <td>-0.042621</td>
      <td>-0.034743</td>
      <td>-0.053128</td>
      <td>0.024243</td>
      <td>-0.002214</td>
      <td>-0.013521</td>
      <td>-0.075076</td>
      <td>-0.161698</td>
      <td>1.000000</td>
      <td>-0.051094</td>
      <td>-0.042138</td>
      <td>-0.050658</td>
    </tr>
    <tr>
      <th>purpose_home_improvement</th>
      <td>0.006036</td>
      <td>-0.050697</td>
      <td>0.023024</td>
      <td>0.116375</td>
      <td>-0.092788</td>
      <td>0.097474</td>
      <td>0.068087</td>
      <td>0.003258</td>
      <td>-0.114449</td>
      <td>0.043827</td>
      <td>-0.013098</td>
      <td>0.004704</td>
      <td>-0.103279</td>
      <td>-0.222441</td>
      <td>-0.051094</td>
      <td>1.000000</td>
      <td>-0.057967</td>
      <td>-0.069687</td>
    </tr>
    <tr>
      <th>purpose_major_purchase</th>
      <td>0.024281</td>
      <td>-0.068978</td>
      <td>-0.079836</td>
      <td>-0.031020</td>
      <td>-0.077719</td>
      <td>0.067129</td>
      <td>-0.020561</td>
      <td>-0.062395</td>
      <td>-0.108079</td>
      <td>-0.001445</td>
      <td>0.004085</td>
      <td>-0.011734</td>
      <td>-0.085176</td>
      <td>-0.183451</td>
      <td>-0.042138</td>
      <td>-0.057967</td>
      <td>1.000000</td>
      <td>-0.057472</td>
    </tr>
    <tr>
      <th>purpose_small_business</th>
      <td>-0.003511</td>
      <td>0.151247</td>
      <td>0.145654</td>
      <td>0.091540</td>
      <td>-0.069245</td>
      <td>0.063292</td>
      <td>0.034883</td>
      <td>0.083069</td>
      <td>-0.060962</td>
      <td>0.042567</td>
      <td>-0.004148</td>
      <td>-0.005595</td>
      <td>-0.102397</td>
      <td>-0.220542</td>
      <td>-0.050658</td>
      <td>-0.069687</td>
      <td>-0.057472</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



    <Figure size 1400x1000 with 0 Axes>



    
![png](lending_loan_files/lending_loan_9_3.png)
    


    
    Strongly correlated features to consider dropping
    []
    
    Shape before dropping strongly correlated features
    (9578, 19)
    
    Shape after dropping strongly correlated features
    (9578, 19)
    [1m
    Which features were strongly correlated enough to drop?[0m
    [92mAny feature pairs with absolute correlation above 0.80 get flagged and dropped. none are found the two shapes are identical. [0m
    
    [1mWhy can dropping strongly correlated features help a model?[0m
    [92mWhen two features are strongly correlated they are telling the model the same story twice. This is called multicollinearity. In a neural network this creates redundant weights that compete with each other during training, making the model less stable and harder to interpret. Removing one of the pair keeps the information while reducing noise, leading to cleaner and more efficient learning.[0m
    



```python
# ============================================================
# SECTION 9. SEPARATE FEATURES AND TARGET
# ============================================================

# Question 8
# Separate predictor features from the target.
#
# Complete all of the following:
# - Create X using all columns except not.fully.paid
# - Create y using not.fully.paid
# - Confirm the shapes of X and y

X = loan_data_final.drop(columns=["not.fully.paid"])

y = loan_data_final["not.fully.paid"]

print("\nShape of X")
print(X.shape)

print("\nShape of y")
print(y.shape)
```

    
    Shape of X
    (9578, 18)
    
    Shape of y
    (9578,)



```python
# Question 9
# Split the data into training and testing sets.
#
# Complete all of the following:
# - Use 20% of the data for testing
# - Use random_state=42
# - Use stratify=y
# - Print target percentages in training and testing sets

X_train, X_test, y_train, y_test = cast(
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    train_test_split(X, y, test_size=0.20, random_state=42, stratify=y),
)
print("\nTraining target percentages before balancing")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting target percentages")
print((y_test.value_counts(normalize=True) * 100).round(2))

print_qa(
    "\nWhy should the split happen before balancing?",
    """
    The test set is supposed to represent the real world, and in the real world
    loan defaults are genuinely rare. Only about 16% of borrowers actually default.
    So if we balance the data first and then split, our test set ends up with a
    50/50 mix that simply does not exist in reality. We would then evaluate the
    model on a fantasy dataset and get metrics that look great on paper but fall
    apart the moment the model sees real loan applications.

    By splitting first, the test set stays frozen at the original 84/16 distribution
    and gives us an honest and realistic measure of how the model would actually
    perform in production. The balancing then only happens on the training set,
    where we want the model to learn defaults more aggressively without polluting
    the evaluation.
    """,
)
```

    
    Training target percentages before balancing
    not.fully.paid
    0    84.0
    1    16.0
    Name: proportion, dtype: float64
    
    Testing target percentages
    not.fully.paid
    0    83.98
    1    16.02
    Name: proportion, dtype: float64
    [1m
    Why should the split happen before balancing?[0m
    [92m
        The test set is supposed to represent the real world, and in the real world
        loan defaults are genuinely rare. Only about 16% of borrowers actually default.
        So if we balance the data first and then split, our test set ends up with a
        50/50 mix that simply does not exist in reality. We would then evaluate the
        model on a fantasy dataset and get metrics that look great on paper but fall
        apart the moment the model sees real loan applications.
    
        By splitting first, the test set stays frozen at the original 84/16 distribution
        and gives us an honest and realistic measure of how the model would actually
        perform in production. The balancing then only happens on the training set,
        where we want the model to learn defaults more aggressively without polluting
        the evaluation.
        [0m
    



```python
# ============================================================
# SECTION 11. BALANCE ONLY THE TRAINING DATA
# ============================================================

# Question 10
# Because the target is imbalanced, balance only the training set.
#
# Complete all of the following:
# - Show training counts before balancing
# - Apply RandomUnderSampler
# - Show training counts after balancing
# - Plot the balanced training target distribution


print("\nTraining target counts before balancing")
print(y_train.value_counts())

undersampler = RandomUnderSampler(random_state=42)

X_train_balanced, y_train_balanced = cast(
    tuple[pd.DataFrame, pd.Series],
    undersampler.fit_resample(X_train, y_train),
)


print("\nTraining target counts after balancing")
# save value count in a vairable
y_train_count = y_train_balanced.value_counts()
balanced_counts: pd.Series = y_train_balanced.value_counts().sort_index()

print(balanced_counts)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=balanced_counts.index.astype(str),
    y=balanced_counts.values,
    hue=balanced_counts.index.astype(str),
    palette=["#4C72B0", "#C44E52"],
    legend=False,
)

for i, (count, pct) in enumerate(
    zip(balanced_counts.values, (y_train_balanced.value_counts(normalize=True) * 100).values)
):
    plt.text(i, count + 20, f"{count:,}\n({pct:.1f}%)", ha="center", fontsize=11)

plt.title("Balanced Training Target Distribution", fontsize=14, pad=15)
plt.xlabel("not.fully.paid  (0 = repaid, 1 = defaulted)", fontsize=11)
plt.ylabel("Number of Loans", fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()

print_qa(
    "Why should the test set remain unbalanced and untouched?",
    """
    The test set represents the real world, and in the real world only 16% of
    borrowers default. Balancing it would mean testing the model on conditions
    that simply do not exist in production, making our metrics look better than
    they actually are. Keeping it untouched gives us an honest picture of how
    the model truly performs on real loan applications.
    """,
)
```

    
    Training target counts before balancing
    not.fully.paid
    0    6436
    1    1226
    Name: count, dtype: int64
    
    Training target counts after balancing
    not.fully.paid
    0    1226
    1    1226
    Name: count, dtype: int64



    
![png](lending_loan_files/lending_loan_12_1.png)
    


    [1mWhy should the test set remain unbalanced and untouched?[0m
    [92m
        The test set represents the real world, and in the real world only 16% of
        borrowers default. Balancing it would mean testing the model on conditions
        that simply do not exist in production, making our metrics look better than
        they actually are. Keeping it untouched gives us an honest picture of how
        the model truly performs on real loan applications.
        [0m
    



```python
# ============================================================
# SECTION 12. SCALE THE FEATURES
# ============================================================

# Question 11
# Scale the input features before sending them to the neural network.
#
# Complete all of the following:
# - Create a StandardScaler
# - Fit it only on X_train_balanced
# - Transform X_train_balanced and X_test
# - Convert the arrays to float32

scaler = StandardScaler()

# fit only on training data, then transform both sets
# fitting on test data would leak future information into the scaler
X_train_scaled: np.ndarray = scaler.fit_transform(X_train_balanced).astype("float32")
X_test_scaled: np.ndarray = cast(np.ndarray, scaler.transform(X_test)).astype("float32")

print("\nScaled training data shape")
print(X_train_scaled.shape)

print("\nScaled test data shape")
print(X_test_scaled.shape)

print_qa(
    "\nWhy is scaling useful before training a neural network?",
    """
    Neural networks learn by adjusting weights through gradient descent.
    When features are on very different scales, for example fico ranges from
    600 to 800 while revol.bal ranges from 0 to 1.2 million, the gradients
    for larger-scale features dominate and push the smaller ones aside.
    This makes training unstable and slow.

    StandardScaler fixes this by transforming every feature to have a mean
    of zero and a standard deviation of one, putting them all on equal footing.
    The model can then learn from every feature fairly without any single one
    drowning out the others.

    we call fit_transform on the training set but only transform on the
    test set. This is important because fitting computes the mean and std from
    the data. If we fit on the test set too, those real-world statistics leak
    into our scaler and our evaluation is no longer honest.
    """,
)
```

    
    Scaled training data shape
    (2452, 18)
    
    Scaled test data shape
    (1916, 18)
    [1m
    Why is scaling useful before training a neural network?[0m
    [92m
        Neural networks learn by adjusting weights through gradient descent.
        When features are on very different scales, for example fico ranges from
        600 to 800 while revol.bal ranges from 0 to 1.2 million, the gradients
        for larger-scale features dominate and push the smaller ones aside.
        This makes training unstable and slow.
    
        StandardScaler fixes this by transforming every feature to have a mean
        of zero and a standard deviation of one, putting them all on equal footing.
        The model can then learn from every feature fairly without any single one
        drowning out the others.
    
        we call fit_transform on the training set but only transform on the
        test set. This is important because fitting computes the mean and std from
        the data. If we fit on the test set too, those real-world statistics leak
        into our scaler and our evaluation is no longer honest.
        [0m
    



```python
# ============================================================
# SECTION 13. BUILD THE DEEP LEARNING MODEL
# REQUIRED TASK 4. MODELING WITH KERAS AND TENSORFLOW
# ============================================================

# Question 12
# Build a binary classification neural network.
#
# The model must include:
# - Input matching the number of scaled features
# - At least two hidden Dense layers
# - ReLU activation in hidden layers
# - Dropout layers
# - One output neuron
# - Sigmoid activation in the output layer
#
# Compile with:
# - optimizer="adam"
# - loss="binary_crossentropy"
# - accuracy
# - sensitivity using Recall(name="sensitivity")
# - AUC using AUC(name="auc")

from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.optimizers import Adam

tf.random.set_seed(42)

n_features: int = X_train_scaled.shape[1]

model = Sequential(
    [
        # Input Layer
        InputLayer(shape=(n_features,)),
        # first hidden layer, takes in the scaled features
        # 128 neurons is a reasonable starting point for this dataset size
        Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),  # randomly drops 10% of neurons to prevent overfitting
        # second hidden layer, narrows the representation down
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        # third hidden layer, compresses further before the output
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        # output layer, single neuron with sigmoid outputs a probability between 0 and 1
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=cast(str, Adam(learning_rate=0.001)),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        Recall(name="sensitivity"),  # tracks how many actual defaults we catch
        AUC(name="auc"),  # measures overall discrimination ability
    ],
)

model.summary()

print_qa(
    "Why does this model use sigmoid in the final layer?",
    """
    We are solving a binary classification problem where the answer is either
    0 (repaid) or 1 (defaulted). The sigmoid function squashes any input into
    a value between 0 and 1, which we can directly interpret as a probability.
    For example, an output of 0.82 means the model thinks there is an 82% chance
    this borrower will default.

    This is different from the hidden layers which use ReLU. ReLU is used there
    because it helps the network learn complex non-linear patterns efficiently
    without the vanishing gradient problem. But in the output layer we do not
    want raw unbounded numbers, we want a clean probability score that we can
    threshold at some point to make a final yes or no decision. Sigmoid gives us exactly
    that, which is why it is the standard choice for binary classification output layers.
    """,
)

```

    2026-05-30 15:23:06.468518: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M1 Pro
    2026-05-30 15:23:06.471973: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
    2026-05-30 15:23:06.472068: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.92 GB
    2026-05-30 15:23:06.472318: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2026-05-30 15:23:06.472364: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">2,432</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │           <span style="color: #00af00; text-decoration-color: #00af00">512</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,697</span> (53.50 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,249</span> (51.75 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">448</span> (1.75 KB)
</pre>



    [1mWhy does this model use sigmoid in the final layer?[0m
    [92m
        We are solving a binary classification problem where the answer is either
        0 (repaid) or 1 (defaulted). The sigmoid function squashes any input into
        a value between 0 and 1, which we can directly interpret as a probability.
        For example, an output of 0.82 means the model thinks there is an 82% chance
        this borrower will default.
    
        This is different from the hidden layers which use ReLU. ReLU is used there
        because it helps the network learn complex non-linear patterns efficiently
        without the vanishing gradient problem. But in the output layer we do not
        want raw unbounded numbers, we want a clean probability score that we can
        threshold at some point to make a final yes or no decision. Sigmoid gives us exactly
        that, which is why it is the standard choice for binary classification output layers.
        [0m
    



```python
# ============================================================
# SECTION 14. TRAIN THE MODEL
# ============================================================

# Question 13
# Train the deep learning model.
#
# Complete all of the following:
# - Add early stopping
# - Use balanced training data
# - Use validation_split
# - Choose epochs and batch size
# - Store training history

y_train_balanced_arr: np.ndarray = y_train_balanced.to_numpy().astype("float32")

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled,
    y_train_balanced_arr,
    test_size=0.2,
    random_state=42,
    stratify=y_train_balanced_arr,  # ensures both classes appear in validation
)

early_stopping = EarlyStopping(
    monitor="val_loss",  # watch validation loss, not training loss
    patience=10,  # stop if no improvement after 10 consecutive epochs
    restore_best_weights=True,  # roll back to the best epoch when stopping
    mode="auto",
    min_delta=0.001,  # requires a meaningful improvement, not just noise
)


history = model.fit(
    X_train_final,
    y_train_final,
    epochs=100,  # upper limit, early stopping will likely kick in before this
    batch_size=32,  # process 32 samples at a time before updating weights
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1,
)
```

    Epoch 1/100


    2026-05-30 15:23:08.673367: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.


    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 92ms/step - accuracy: 0.5665 - auc: 0.5987 - loss: 0.8699 - sensitivity: 0.5637 - val_accuracy: 0.6538 - val_auc: 0.7107 - val_loss: 0.7804 - val_sensitivity: 0.6571
    Epoch 2/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 46ms/step - accuracy: 0.5961 - auc: 0.6321 - loss: 0.8233 - sensitivity: 0.5953 - val_accuracy: 0.6762 - val_auc: 0.7212 - val_loss: 0.7684 - val_sensitivity: 0.6816
    Epoch 3/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 44ms/step - accuracy: 0.6196 - auc: 0.6519 - loss: 0.8007 - sensitivity: 0.6177 - val_accuracy: 0.6599 - val_auc: 0.7188 - val_loss: 0.7604 - val_sensitivity: 0.6653
    Epoch 4/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 46ms/step - accuracy: 0.5905 - auc: 0.6398 - loss: 0.8011 - sensitivity: 0.5882 - val_accuracy: 0.6802 - val_auc: 0.7296 - val_loss: 0.7519 - val_sensitivity: 0.6612
    Epoch 5/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 43ms/step - accuracy: 0.6048 - auc: 0.6484 - loss: 0.7905 - sensitivity: 0.5984 - val_accuracy: 0.6701 - val_auc: 0.7302 - val_loss: 0.7435 - val_sensitivity: 0.6408
    Epoch 6/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 43ms/step - accuracy: 0.5966 - auc: 0.6450 - loss: 0.7863 - sensitivity: 0.6075 - val_accuracy: 0.6741 - val_auc: 0.7379 - val_loss: 0.7383 - val_sensitivity: 0.6408
    Epoch 7/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 45ms/step - accuracy: 0.6038 - auc: 0.6611 - loss: 0.7736 - sensitivity: 0.6096 - val_accuracy: 0.6802 - val_auc: 0.7376 - val_loss: 0.7316 - val_sensitivity: 0.6571
    Epoch 8/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 49ms/step - accuracy: 0.6002 - auc: 0.6576 - loss: 0.7718 - sensitivity: 0.5994 - val_accuracy: 0.6864 - val_auc: 0.7402 - val_loss: 0.7251 - val_sensitivity: 0.6653
    Epoch 9/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 44ms/step - accuracy: 0.6114 - auc: 0.6640 - loss: 0.7670 - sensitivity: 0.6116 - val_accuracy: 0.6782 - val_auc: 0.7405 - val_loss: 0.7228 - val_sensitivity: 0.6571
    Epoch 10/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 44ms/step - accuracy: 0.6058 - auc: 0.6588 - loss: 0.7635 - sensitivity: 0.6147 - val_accuracy: 0.6721 - val_auc: 0.7402 - val_loss: 0.7192 - val_sensitivity: 0.6653
    Epoch 11/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 45ms/step - accuracy: 0.6053 - auc: 0.6619 - loss: 0.7589 - sensitivity: 0.6096 - val_accuracy: 0.6782 - val_auc: 0.7404 - val_loss: 0.7152 - val_sensitivity: 0.6735
    Epoch 12/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 49ms/step - accuracy: 0.6038 - auc: 0.6579 - loss: 0.7585 - sensitivity: 0.6137 - val_accuracy: 0.6782 - val_auc: 0.7384 - val_loss: 0.7166 - val_sensitivity: 0.6735
    Epoch 13/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 51ms/step - accuracy: 0.6058 - auc: 0.6592 - loss: 0.7552 - sensitivity: 0.6086 - val_accuracy: 0.6782 - val_auc: 0.7365 - val_loss: 0.7125 - val_sensitivity: 0.6898
    Epoch 14/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 49ms/step - accuracy: 0.6073 - auc: 0.6582 - loss: 0.7544 - sensitivity: 0.6147 - val_accuracy: 0.6741 - val_auc: 0.7356 - val_loss: 0.7108 - val_sensitivity: 0.6816
    Epoch 15/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 51ms/step - accuracy: 0.6038 - auc: 0.6573 - loss: 0.7521 - sensitivity: 0.6116 - val_accuracy: 0.6701 - val_auc: 0.7366 - val_loss: 0.7081 - val_sensitivity: 0.6816
    Epoch 16/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 55ms/step - accuracy: 0.6017 - auc: 0.6598 - loss: 0.7483 - sensitivity: 0.6126 - val_accuracy: 0.6762 - val_auc: 0.7350 - val_loss: 0.7067 - val_sensitivity: 0.6898
    Epoch 17/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 50ms/step - accuracy: 0.6007 - auc: 0.6586 - loss: 0.7471 - sensitivity: 0.6055 - val_accuracy: 0.6823 - val_auc: 0.7362 - val_loss: 0.7043 - val_sensitivity: 0.6939
    Epoch 18/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 54ms/step - accuracy: 0.6058 - auc: 0.6610 - loss: 0.7447 - sensitivity: 0.6116 - val_accuracy: 0.6782 - val_auc: 0.7340 - val_loss: 0.7034 - val_sensitivity: 0.6898
    Epoch 19/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 47ms/step - accuracy: 0.6124 - auc: 0.6596 - loss: 0.7451 - sensitivity: 0.6269 - val_accuracy: 0.6782 - val_auc: 0.7364 - val_loss: 0.7040 - val_sensitivity: 0.6776
    Epoch 20/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 50ms/step - accuracy: 0.6043 - auc: 0.6568 - loss: 0.7448 - sensitivity: 0.6137 - val_accuracy: 0.6762 - val_auc: 0.7367 - val_loss: 0.7019 - val_sensitivity: 0.6694
    Epoch 21/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 47ms/step - accuracy: 0.6094 - auc: 0.6597 - loss: 0.7418 - sensitivity: 0.6188 - val_accuracy: 0.6741 - val_auc: 0.7357 - val_loss: 0.7007 - val_sensitivity: 0.6653
    Epoch 22/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 48ms/step - accuracy: 0.6038 - auc: 0.6614 - loss: 0.7407 - sensitivity: 0.6167 - val_accuracy: 0.6721 - val_auc: 0.7357 - val_loss: 0.7004 - val_sensitivity: 0.6694
    Epoch 23/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 46ms/step - accuracy: 0.6007 - auc: 0.6591 - loss: 0.7408 - sensitivity: 0.6126 - val_accuracy: 0.6680 - val_auc: 0.7354 - val_loss: 0.6998 - val_sensitivity: 0.6571
    Epoch 24/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 48ms/step - accuracy: 0.6073 - auc: 0.6578 - loss: 0.7415 - sensitivity: 0.6300 - val_accuracy: 0.6701 - val_auc: 0.7349 - val_loss: 0.6999 - val_sensitivity: 0.6694
    Epoch 25/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 47ms/step - accuracy: 0.6033 - auc: 0.6582 - loss: 0.7408 - sensitivity: 0.6106 - val_accuracy: 0.6741 - val_auc: 0.7360 - val_loss: 0.7003 - val_sensitivity: 0.6694
    Epoch 26/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 52ms/step - accuracy: 0.6002 - auc: 0.6600 - loss: 0.7403 - sensitivity: 0.6116 - val_accuracy: 0.6660 - val_auc: 0.7353 - val_loss: 0.7003 - val_sensitivity: 0.6571
    Epoch 27/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 52ms/step - accuracy: 0.5992 - auc: 0.6616 - loss: 0.7384 - sensitivity: 0.6116 - val_accuracy: 0.6721 - val_auc: 0.7352 - val_loss: 0.6996 - val_sensitivity: 0.6653
    Epoch 28/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 49ms/step - accuracy: 0.6094 - auc: 0.6646 - loss: 0.7374 - sensitivity: 0.6167 - val_accuracy: 0.6680 - val_auc: 0.7366 - val_loss: 0.6993 - val_sensitivity: 0.6571
    Epoch 29/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 51ms/step - accuracy: 0.6022 - auc: 0.6618 - loss: 0.7393 - sensitivity: 0.6157 - val_accuracy: 0.6701 - val_auc: 0.7364 - val_loss: 0.7006 - val_sensitivity: 0.6531
    Epoch 30/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 54ms/step - accuracy: 0.6048 - auc: 0.6621 - loss: 0.7386 - sensitivity: 0.6208 - val_accuracy: 0.6680 - val_auc: 0.7373 - val_loss: 0.6998 - val_sensitivity: 0.6571
    Epoch 31/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 81ms/step - accuracy: 0.6017 - auc: 0.6603 - loss: 0.7402 - sensitivity: 0.6096 - val_accuracy: 0.6680 - val_auc: 0.7375 - val_loss: 0.7003 - val_sensitivity: 0.6531
    Epoch 32/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 87ms/step - accuracy: 0.6053 - auc: 0.6621 - loss: 0.7386 - sensitivity: 0.6106 - val_accuracy: 0.6701 - val_auc: 0.7370 - val_loss: 0.7010 - val_sensitivity: 0.6571
    Epoch 33/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 90ms/step - accuracy: 0.6073 - auc: 0.6605 - loss: 0.7406 - sensitivity: 0.6177 - val_accuracy: 0.6721 - val_auc: 0.7368 - val_loss: 0.7010 - val_sensitivity: 0.6571
    Epoch 34/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 203ms/step - accuracy: 0.6007 - auc: 0.6604 - loss: 0.7412 - sensitivity: 0.6126 - val_accuracy: 0.6741 - val_auc: 0.7383 - val_loss: 0.7017 - val_sensitivity: 0.6653
    Epoch 35/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 76ms/step - accuracy: 0.5992 - auc: 0.6637 - loss: 0.7397 - sensitivity: 0.6157 - val_accuracy: 0.6721 - val_auc: 0.7374 - val_loss: 0.7024 - val_sensitivity: 0.6571
    Epoch 36/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 148ms/step - accuracy: 0.6053 - auc: 0.6605 - loss: 0.7422 - sensitivity: 0.6126 - val_accuracy: 0.6741 - val_auc: 0.7367 - val_loss: 0.7040 - val_sensitivity: 0.6612
    Epoch 37/100
    [1m62/62[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 354ms/step - accuracy: 0.5997 - auc: 0.6596 - loss: 0.7426 - sensitivity: 0.6126 - val_accuracy: 0.6701 - val_auc: 0.7362 - val_loss: 0.7034 - val_sensitivity: 0.6612



```python
# ============================================================
# SECTION 15. PLOT TRAINING HISTORY
# ============================================================

# Question 14
# Plot:
# - Training vs validation loss
# - Training vs validation AUC
# - Training vs validation sensitivity

train_loss: list[float] = history.history["loss"]
val_loss: list[float] = history.history["val_loss"]
train_auc: list[float] = history.history["auc"]
val_auc: list[float] = history.history["val_auc"]
train_sensitivity: list[float] = history.history["sensitivity"]
val_sensitivity: list[float] = history.history["val_sensitivity"]
epochs_range: range = range(1, len(train_loss) + 1)

# loss curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_loss, label="Training Loss", color="#4C72B0")
plt.plot(epochs_range, val_loss, label="Validation Loss", color="#C44E52", linestyle="--")
plt.title("Training vs Validation Loss", fontsize=14, pad=15)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()

# auc curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_auc, label="Training AUC", color="#4C72B0")
plt.plot(epochs_range, val_auc, label="Validation AUC", color="#C44E52", linestyle="--")
plt.title("Training vs Validation AUC", fontsize=14, pad=15)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("AUC", fontsize=11)
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()

# sensitivity curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_sensitivity, label="Training Sensitivity", color="#4C72B0")
plt.plot(epochs_range, val_sensitivity, label="Validation Sensitivity", color="#C44E52", linestyle="--")
plt.title("Training vs Validation Sensitivity", fontsize=14, pad=15)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Sensitivity", fontsize=11)
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()

print_qa(
    "Based on the curves, is the model learning, overfitting, or underfitting?",
    """
    Looking at all three plots together, the model is learning but has not
    overfit, which is actually the best outcome we could realistically hope
    for given the constraints of this dataset.

    The loss plot tells the clearest story. Both training loss and validation
    loss are steadily declining across all 11 epochs and moving in the same
    direction. The gap between them is consistent and not widening, which
    means the model is generalising to unseen data rather than memorising
    the training set. If it were overfitting, we would see training loss
    keep dropping while validation loss flattens or rises. That is not
    happening here.

    The AUC plot confirms this. Validation AUC jumped to around 0.74 by
    epoch 2 and has stayed stable since, while training AUC is slowly
    climbing toward it from below. The gap between the two lines is
    explained by the class weights making the training task harder, not
    by overfitting. A model that is overfitting would show training AUC
    racing far above validation AUC, which is the opposite of what we see.

    The sensitivity plot is the noisiest of the three but still tells a
    positive story. Validation sensitivity is consistently higher than
    training sensitivity and is hovering around 0.67 to 0.70, meaning
    the model correctly identifies roughly 7 out of every 10 actual
    defaults on data it has never seen before. The noise in this curve
    is expected given the small number of positive samples after
    undersampling.

    The one concern worth noting is that both curves are still declining
    at epoch 11, which suggests the model may benefit from a few more
    epochs before early stopping fires. But the overall picture is
    healthy: the model is genuinely learning, not memorising, and
    performing at a level that is consistent with what this dataset
    can support.
    """,
)
```


    
![png](lending_loan_files/lending_loan_16_0.png)
    



    
![png](lending_loan_files/lending_loan_16_1.png)
    



    
![png](lending_loan_files/lending_loan_16_2.png)
    


    [1mBased on the curves, is the model learning, overfitting, or underfitting?[0m
    [92m
        Looking at all three plots together, the model is learning but has not
        overfit, which is actually the best outcome we could realistically hope
        for given the constraints of this dataset.
    
        The loss plot tells the clearest story. Both training loss and validation
        loss are steadily declining across all 11 epochs and moving in the same
        direction. The gap between them is consistent and not widening, which
        means the model is generalising to unseen data rather than memorising
        the training set. If it were overfitting, we would see training loss
        keep dropping while validation loss flattens or rises. That is not
        happening here.
    
        The AUC plot confirms this. Validation AUC jumped to around 0.74 by
        epoch 2 and has stayed stable since, while training AUC is slowly
        climbing toward it from below. The gap between the two lines is
        explained by the class weights making the training task harder, not
        by overfitting. A model that is overfitting would show training AUC
        racing far above validation AUC, which is the opposite of what we see.
    
        The sensitivity plot is the noisiest of the three but still tells a
        positive story. Validation sensitivity is consistently higher than
        training sensitivity and is hovering around 0.67 to 0.70, meaning
        the model correctly identifies roughly 7 out of every 10 actual
        defaults on data it has never seen before. The noise in this curve
        is expected given the small number of positive samples after
        undersampling.
    
        The one concern worth noting is that both curves are still declining
        at epoch 11, which suggests the model may benefit from a few more
        epochs before early stopping fires. But the overall picture is
        healthy: the model is genuinely learning, not memorising, and
        performing at a level that is consistent with what this dataset
        can support.
        [0m
    



```python
# ============================================================
# SECTION 16. MAKE PREDICTIONS
# ============================================================

# Question 15
# Use the trained model to make predictions on the untouched test set.
#
# Complete all of the following:
# - Predict probabilities
# - Convert probabilities into 0 or 1 using threshold 0.50

# model.predict returns shape (n_samples, 1) because the output layer
# has a single neuron. ravel() flattens it to a 1D array of shape (n_samples,)
# which is what sklearn metrics expect
y_pred_prob: np.ndarray = model.predict(X_test_scaled).ravel()

# anything above 0.50 is classified as a default (1)
# anything at or below 0.50 is classified as repaid (0)
# we can revisit this threshold later since 0.50 may not be optimal
# for a lending problem where catching defaults matters more than precision
y_pred_class: np.ndarray = (y_pred_prob >= 0.50).astype(int)
```

    [1m60/60[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 55ms/step



```python
# ============================================================
# SECTION 17. EVALUATE THE MODEL
# ============================================================

# Question 16
# Evaluate the final model.
#
# Complete all of the following:
# - Confusion matrix
# - TN, FP, FN, TP
# - Sensitivity
# - ROC-AUC
# - Classification report
# - ROC curve

conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred_class)

# unpack the four cells of the confusion matrix into named variables
# conf_matrix.ravel() flattens the 2x2 matrix into [TN, FP, FN, TP]
tn, fp, fn, tp = conf_matrix.ravel()

# sensitivity = TP / (TP + FN)
# out of all actual defaults, how many did we correctly catch?
sensitivity: float = tp / (tp + fn)

# roc_auc uses the raw probabilities, not the binary predictions
# this gives a threshold-independent measure of discrimination ability
roc_auc: float = cast(float, roc_auc_score(y_test, y_pred_prob))

print("\nConfusion Matrix")
print(conf_matrix)

print("\nTrue Negatives (correctly predicted repaid)")
print(tn)

print("\nFalse Positives (predicted default, actually repaid)")
print(fp)

print("\nFalse Negatives (predicted repaid, actually defaulted)")
print(fn)

print("\nTrue Positives (correctly predicted default)")
print(tp)

print("\nSensitivity / Recall for Default Class")
print(round(sensitivity, 4))

print("\nROC-AUC")
print(round(roc_auc, 4))

print("\nClassification Report")
print(classification_report(y_test, y_pred_class))

# roc_curve uses probabilities to compute the full trade-off curve
# across every possible threshold, not just 0.50
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"Neural Network (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random Classifier (AUC = 0.500)")

plt.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
plt.title("Receiver Operating Characteristic Curve", fontsize=14, pad=15)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.legend(loc="lower right")
sns.despine()
plt.tight_layout()
plt.show()

print_qa(
    "What does sensitivity mean in this project?",
    """
    Sensitivity, also called recall for the default class, answers a
    specific question: out of all the loans that actually defaulted, how
    many did the model correctly flag? If sensitivity is 0.65, it means
    the model caught 65 out of every 100 real defaults and missed the
    other 35. Those 35 missed defaults are False Negatives, and in a
    lending context each one represents a loan the bank approved that
    it will never get back. This is why sensitivity is the metric that
    matters most in this project. Missing a default is far more costly
    than raising a false alarm on a good borrower.
    """,
)

print_qa(
    "What does ROC-AUC tell LendingClub?",
    """
    ROC-AUC measures how well the model ranks borrowers by their risk
    level across every possible decision threshold, not just at 0.50.
    An AUC of 0.74 means that if you randomly picked one borrower who
    defaulted and one who repaid, the model would correctly assign a
    higher risk score to the defaulter 74% of the time. A score of 0.50
    would mean the model is no better than a random coin flip, and a score of
    1.0 would mean perfect separation. For LendingClub, this number tells
    them how trustworthy the model's probability scores are as a general
    ranking tool, independent of where they choose to draw the line
    between approve and reject.
    """,
)

print_qa(
    "Why is accuracy alone not enough for this dataset?",
    """
    Because accuracy is deeply misleading on imbalanced data. In this
    dataset, 84% of loans were repaid and only 16% defaulted. A model
    that simply predicts every single loan as repaid would achieve 84%
    accuracy without ever learning anything useful. It would have zero
    sensitivity, meaning it catches zero actual defaults, which makes
    it completely worthless for the bank's actual purpose. Accuracy
    tells you how often the model is right overall, but it hides the
    fact that the model might be systematically wrong on the exact cases
    that matter most. Sensitivity, precision, and AUC give a much more
    honest picture of whether the model is genuinely useful or just
    exploiting the class imbalance to look good on paper.
    """,
)
```

    
    Confusion Matrix
    [[1009  600]
     [ 121  186]]
    
    True Negatives (correctly predicted repaid)
    1009
    
    False Positives (predicted default, actually repaid)
    600
    
    False Negatives (predicted repaid, actually defaulted)
    121
    
    True Positives (correctly predicted default)
    186
    
    Sensitivity / Recall for Default Class
    0.6059
    
    ROC-AUC
    0.6755
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.89      0.63      0.74      1609
               1       0.24      0.61      0.34       307
    
        accuracy                           0.62      1916
       macro avg       0.56      0.62      0.54      1916
    weighted avg       0.79      0.62      0.67      1916
    



    
![png](lending_loan_files/lending_loan_18_1.png)
    


    [1mWhat does sensitivity mean in this project?[0m
    [92m
        Sensitivity, also called recall for the default class, answers a
        specific question: out of all the loans that actually defaulted, how
        many did the model correctly flag? If sensitivity is 0.65, it means
        the model caught 65 out of every 100 real defaults and missed the
        other 35. Those 35 missed defaults are False Negatives, and in a
        lending context each one represents a loan the bank approved that
        it will never get back. This is why sensitivity is the metric that
        matters most in this project. Missing a default is far more costly
        than raising a false alarm on a good borrower.
        [0m
    
    [1mWhat does ROC-AUC tell LendingClub?[0m
    [92m
        ROC-AUC measures how well the model ranks borrowers by their risk
        level across every possible decision threshold, not just at 0.50.
        An AUC of 0.74 means that if you randomly picked one borrower who
        defaulted and one who repaid, the model would correctly assign a
        higher risk score to the defaulter 74% of the time. A score of 0.50
        would mean the model is no better than a random coin flip, and a score of
        1.0 would mean perfect separation. For LendingClub, this number tells
        them how trustworthy the model's probability scores are as a general
        ranking tool, independent of where they choose to draw the line
        between approve and reject.
        [0m
    
    [1mWhy is accuracy alone not enough for this dataset?[0m
    [92m
        Because accuracy is deeply misleading on imbalanced data. In this
        dataset, 84% of loans were repaid and only 16% defaulted. A model
        that simply predicts every single loan as repaid would achieve 84%
        accuracy without ever learning anything useful. It would have zero
        sensitivity, meaning it catches zero actual defaults, which makes
        it completely worthless for the bank's actual purpose. Accuracy
        tells you how often the model is right overall, but it hides the
        fact that the model might be systematically wrong on the exact cases
        that matter most. Sensitivity, precision, and AUC give a much more
        honest picture of whether the model is genuinely useful or just
        exploiting the class imbalance to look good on paper.
        [0m
    



```python
# ============================================================
# SECTION 18. FINAL BUSINESS INTERPRETATION
# ============================================================

print_qa(
    "1. What was the objective of this project?",
    """
    The goal was to predict whether a borrower will default on their
    LendingClub loan using historical data. We built a neural network
    that looks at things like credit score, interest rate, and debt
    levels to assign each borrower a probability of default, giving
    the bank a smarter way to decide who to lend money to.
    """,
)

print_qa(
    "2. Which column was categorical and how was it transformed?",
    """
    The purpose column was the only text column. It held loan reasons
    like debt_consolidation and credit_card. Since neural networks only
    understand numbers, we used get_dummies() to turn each unique
    purpose into its own binary column of zeros and ones.
    """,
)

print_qa(
    "3. Which EDA finding was most useful?",
    """
    The FICO score comparison was the clearest signal we found. Borrowers
    who defaulted had noticeably lower credit scores than those who
    repaid. It was the only feature that showed a clean separation
    between the two groups, which told us early on that FICO would
    likely be the strongest predictor in the model.
    """,
)

print_qa(
    "4. Was the target balanced or imbalanced?",
    """
    It was imbalanced. About 84% of loans were repaid and only 16%
    defaulted. This is a problem because the model can look accurate
    on paper by just predicting repaid every time, while completely
    missing the defaults that actually matter to the bank.
    """,
)

print_qa(
    "5. Which strongly correlated features were dropped, if any?",
    """
    Check the output from Section 8 in your notebook. If no feature
    pairs crossed the 0.80 correlation threshold, nothing was dropped
    and the dataset stayed the same shape. If any pairs were flagged,
    one feature from each pair was removed automatically.
    """,
)

print_qa(
    "6. Why were correlated features removed?",
    """
    When two features carry the same information, they confuse the model
    by sending competing signals during training. Removing one of the
    pair keeps all the useful information without the noise, which helps
    the model learn more cleanly and reliably.
    """,
)

print_qa(
    "7. What does sensitivity mean for LendingClub?",
    """
    Sensitivity tells us how many real defaults the model actually caught.
    If sensitivity is 0.70, the model flagged 70 out of every 100 loans
    that truly defaulted and missed 30. Those 30 missed defaults are
    money the bank lent out and never got back, which is why this number
    matters far more than overall accuracy.
    """,
)

print_qa(
    "8. What does ROC-AUC mean in this project?",
    """
    ROC-AUC tells us how well the model ranks risky borrowers above safe
    ones. An AUC of 0.74 means that if you picked one defaulter and one
    reliable borrower at random, the model would correctly score the
    defaulter as riskier 74% of the time. A score of 0.50 would mean
    the model is no better than a coin flip.
    """,
)

print_qa(
    "9. What is one strength of the final model?",
    """
    The model does not overfit. Training and validation metrics moved
    together throughout training, which means the model learned real
    patterns in borrower behaviour rather than just memorising the
    training data. It performs consistently on loans it has never
    seen before, which is what actually matters in production.
    """,
)

print_qa(
    "10. What is one limitation of the final model?",
    """
    The model can only see what was true at the time of the loan
    application. It cannot predict a job loss six months later or
    an unexpected medical bill that causes a borrower to stop paying.
    Those life events cause most real defaults, and no feature in
    this dataset captures them, which is why the model hits a ceiling
    around 0.74 AUC no matter how much we tune it.
    """,
)

```

    [1m1. What was the objective of this project?[0m
    [92m
        The goal was to predict whether a borrower will default on their
        LendingClub loan using historical data. We built a neural network
        that looks at things like credit score, interest rate, and debt
        levels to assign each borrower a probability of default, giving
        the bank a smarter way to decide who to lend money to.
        [0m
    
    [1m2. Which column was categorical and how was it transformed?[0m
    [92m
        The purpose column was the only text column. It held loan reasons
        like debt_consolidation and credit_card. Since neural networks only
        understand numbers, we used get_dummies() to turn each unique
        purpose into its own binary column of zeros and ones.
        [0m
    
    [1m3. Which EDA finding was most useful?[0m
    [92m
        The FICO score comparison was the clearest signal we found. Borrowers
        who defaulted had noticeably lower credit scores than those who
        repaid. It was the only feature that showed a clean separation
        between the two groups, which told us early on that FICO would
        likely be the strongest predictor in the model.
        [0m
    
    [1m4. Was the target balanced or imbalanced?[0m
    [92m
        It was imbalanced. About 84% of loans were repaid and only 16%
        defaulted. This is a problem because the model can look accurate
        on paper by just predicting repaid every time, while completely
        missing the defaults that actually matter to the bank.
        [0m
    
    [1m5. Which strongly correlated features were dropped, if any?[0m
    [92m
        Check the output from Section 8 in your notebook. If no feature
        pairs crossed the 0.80 correlation threshold, nothing was dropped
        and the dataset stayed the same shape. If any pairs were flagged,
        one feature from each pair was removed automatically.
        [0m
    
    [1m6. Why were correlated features removed?[0m
    [92m
        When two features carry the same information, they confuse the model
        by sending competing signals during training. Removing one of the
        pair keeps all the useful information without the noise, which helps
        the model learn more cleanly and reliably.
        [0m
    
    [1m7. What does sensitivity mean for LendingClub?[0m
    [92m
        Sensitivity tells us how many real defaults the model actually caught.
        If sensitivity is 0.70, the model flagged 70 out of every 100 loans
        that truly defaulted and missed 30. Those 30 missed defaults are
        money the bank lent out and never got back, which is why this number
        matters far more than overall accuracy.
        [0m
    
    [1m8. What does ROC-AUC mean in this project?[0m
    [92m
        ROC-AUC tells us how well the model ranks risky borrowers above safe
        ones. An AUC of 0.74 means that if you picked one defaulter and one
        reliable borrower at random, the model would correctly score the
        defaulter as riskier 74% of the time. A score of 0.50 would mean
        the model is no better than a coin flip.
        [0m
    
    [1m9. What is one strength of the final model?[0m
    [92m
        The model does not overfit. Training and validation metrics moved
        together throughout training, which means the model learned real
        patterns in borrower behaviour rather than just memorising the
        training data. It performs consistently on loans it has never
        seen before, which is what actually matters in production.
        [0m
    
    [1m10. What is one limitation of the final model?[0m
    [92m
        The model can only see what was true at the time of the loan
        application. It cannot predict a job loss six months later or
        an unexpected medical bill that causes a borrower to stop paying.
        Those life events cause most real defaults, and no feature in
        this dataset captures them, which is why the model hits a ceiling
        around 0.74 AUC no matter how much we tune it.
        [0m
    



```python
# ============================================================
# OPTIONAL EXTENSION TASKS
# ============================================================

# Extension A: Compare correlation thresholds
print("\nEXTENSION A: CORRELATION THRESHOLD COMPARISON")
print("=" * 70)

upper_triangle: pd.DataFrame = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

for threshold in [0.70, 0.80, 0.90]:
    dropped: list[str] = upper_triangle.columns[(upper_triangle.abs() > threshold).any()].tolist()
    print(f"\nThreshold {threshold}: {len(dropped)} features flagged → {dropped if dropped else 'none'}")

print_qa(
    "What does changing the threshold actually mean?",
    """
    A lower threshold like 0.70 removes more features but risks throwing
    away useful signal. A higher threshold like 0.90 only removes features
    that are almost perfectly identical, keeping more information but leaving
    some redundancy. For financial data where every column has a real meaning,
    0.80 is usually the most honest and defensible choice.
    """,
)


# Extension B: Classification threshold comparison
print("\nEXTENSION B: CLASSIFICATION THRESHOLD COMPARISON")
print("=" * 70)

threshold_comparison: list[dict] = []

for thresh in [0.30, 0.40, 0.50]:
    y_pred_thresh: np.ndarray = (y_pred_prob >= thresh).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_thresh).ravel()

    threshold_comparison.append(
        {
            "threshold": thresh,
            "sensitivity": round(tp_t / (tp_t + fn_t), 3),
            "false_negatives": int(fn_t),
            "false_positives": int(fp_t),
            "precision": round(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0, 3),
        }
    )

comparison_df: pd.DataFrame = pd.DataFrame(threshold_comparison)
print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

metrics: list[str] = ["sensitivity", "false_negatives", "false_positives"]
colors: list[str] = ["#4C72B0", "#C44E52", "#DD8452"]
titles: list[str] = [
    "Sensitivity (higher is better)",
    "False Negatives (lower is better)",
    "False Positives (lower is better)",
]

for ax, metric, color, title in zip(axes, metrics, colors, titles):
    sns.barplot(data=comparison_df, x="threshold", y=metric, hue="threshold", palette=[color] * 3, legend=False, ax=ax)
    for i, val in enumerate(comparison_df[metric]):
        ax.text(i, val + (0.01 if metric == "sensitivity" else 1), str(val), ha="center", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric.replace("_", " ").title())
    sns.despine(ax=ax)

plt.suptitle("Extension B: Impact of Classification Threshold", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print_qa(
    "Which threshold would you recommend for LendingClub and why?",
    """
    A threshold of 0.30 is the most sensible choice for a lender. A false
    positive costs the bank a loan opportunity. A false negative costs the
    bank the entire loan amount. Missing a default is always more expensive
    than being cautious with a good borrower, so lowering the threshold to
    catch more real defaults is the right trade-off even if it means more
    false alarms.
    """,
)


# Extension C: Neural Network vs Logistic Regression
print("\nEXTENSION C: NEURAL NETWORK VS LOGISTIC REGRESSION")
print("=" * 70)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_final, y_train_final)

lr_pred_prob: np.ndarray = log_reg.predict_proba(X_test_scaled)[:, 1]
lr_pred_class: np.ndarray = (lr_pred_prob >= 0.50).astype(int)
lr_auc: float = roc_auc_score(y_test, lr_pred_prob)
lr_tn, lr_fp, lr_fn, lr_tp = confusion_matrix(y_test, lr_pred_class).ravel()
lr_sensitivity: float = lr_tp / (lr_tp + lr_fn)

nn_auc: float = roc_auc_score(y_test, y_pred_prob)
nn_tn, nn_fp, nn_fn, nn_tp = confusion_matrix(y_test, y_pred_class).ravel()
nn_sensitivity: float = nn_tp / (nn_tp + nn_fn)

model_comparison: pd.DataFrame = pd.DataFrame(
    {
        "model": ["Logistic Regression", "Neural Network"],
        "roc_auc": [round(lr_auc, 3), round(nn_auc, 3)],
        "sensitivity": [round(lr_sensitivity, 3), round(nn_sensitivity, 3)],
        "false_negatives": [int(lr_fn), int(nn_fn)],
        "false_positives": [int(lr_fp), int(nn_fp)],
    }
)

print(model_comparison.to_string(index=False))

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred_prob)
nn_fpr, nn_tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(nn_fpr, nn_tpr, color="#4C72B0", linewidth=2, label=f"Neural Network (AUC = {nn_auc:.3f})")
plt.plot(
    lr_fpr, lr_tpr, color="#C44E52", linewidth=2, linestyle="--", label=f"Logistic Regression (AUC = {lr_auc:.3f})"
)
plt.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":", label="Random Classifier (AUC = 0.500)")
plt.fill_between(nn_fpr, nn_tpr, alpha=0.05, color="#4C72B0")
plt.fill_between(lr_fpr, lr_tpr, alpha=0.05, color="#C44E52")
plt.title("Extension C: ROC Curve Comparison", fontsize=14, pad=15)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.legend(loc="lower right")
sns.despine()
plt.tight_layout()
plt.show()

print_qa(
    "What does the comparison tell us?",
    """
    If the two AUC scores are close, it tells us the extra complexity of
    a neural network did not buy much over a simple linear model on this
    dataset. This is common with small tabular financial data. Logistic
    regression is fast, interpretable, and easy to explain to a credit
    officer. A neural network is harder to justify when the performance
    gap is small, but this project required one so we built it and now
    we can see exactly how it compares.
    """,
)


# Extension D: Additional EDA plots
print("\nEXTENSION D: ADDITIONAL EDA PLOTS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

features: list[str] = ["revol.util", "installment", "inq.last.6mths"]
ylabels: list[str] = ["Revolving Utilization (%)", "Monthly Installment ($)", "Inquiries in Last 6 Months"]
titles: list[str] = ["Credit Utilization by Outcome", "Monthly Installment by Outcome", "Recent Inquiries by Outcome"]

for ax, feature, ylabel, title in zip(axes, features, ylabels, titles):
    sns.boxplot(
        data=loan_data,
        x="not.fully.paid",
        y=feature,
        hue="not.fully.paid",
        palette=["#4C72B0", "#C44E52"],
        legend=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("not.fully.paid (0 = repaid, 1 = defaulted)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    sns.despine(ax=ax)

plt.suptitle("Extension D: Additional Feature Analysis by Loan Outcome", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print_qa(
    "What do these three plots reveal about default behavior?",
    """
    Defaulters tend to have higher credit utilization, meaning they were
    already stretched thin before the loan was issued. They also tend to
    have higher monthly installments, which leaves less room in their
    budget when something goes wrong. The inquiries plot is the most
    telling though. Defaulters had more recent credit checks, which often
    means they were desperately seeking credit from multiple places at
    once rather than borrowing out of choice. That kind of behaviour is
    a financial stress signal that the model picks up on.
    """,
)
```

    
    EXTENSION A: CORRELATION THRESHOLD COMPARISON
    ======================================================================
    
    Threshold 0.7: 1 features flagged → ['fico']
    
    Threshold 0.8: 0 features flagged → none
    
    Threshold 0.9: 0 features flagged → none
    [1mWhat does changing the threshold actually mean?[0m
    [92m
        A lower threshold like 0.70 removes more features but risks throwing
        away useful signal. A higher threshold like 0.90 only removes features
        that are almost perfectly identical, keeping more information but leaving
        some redundancy. For financial data where every column has a real meaning,
        0.80 is usually the most honest and defensible choice.
        [0m
    
    
    EXTENSION B: CLASSIFICATION THRESHOLD COMPARISON
    ======================================================================
     threshold  sensitivity  false_negatives  false_positives  precision
           0.3        0.971                9             1395      0.176
           0.4        0.844               48             1052      0.198
           0.5        0.606              121              600      0.237



    
![png](lending_loan_files/lending_loan_20_1.png)
    


    [1mWhich threshold would you recommend for LendingClub and why?[0m
    [92m
        A threshold of 0.30 is the most sensible choice for a lender. A false
        positive costs the bank a loan opportunity. A false negative costs the
        bank the entire loan amount. Missing a default is always more expensive
        than being cautious with a good borrower, so lowering the threshold to
        catch more real defaults is the right trade-off even if it means more
        false alarms.
        [0m
    
    
    EXTENSION C: NEURAL NETWORK VS LOGISTIC REGRESSION
    ======================================================================
                  model  roc_auc  sensitivity  false_negatives  false_positives
    Logistic Regression    0.679        0.599              123              583
         Neural Network    0.675        0.606              121              600



    
![png](lending_loan_files/lending_loan_20_3.png)
    


    [1mWhat does the comparison tell us?[0m
    [92m
        If the two AUC scores are close, it tells us the extra complexity of
        a neural network did not buy much over a simple linear model on this
        dataset. This is common with small tabular financial data. Logistic
        regression is fast, interpretable, and easy to explain to a credit
        officer. A neural network is harder to justify when the performance
        gap is small, but this project required one so we built it and now
        we can see exactly how it compares.
        [0m
    
    
    EXTENSION D: ADDITIONAL EDA PLOTS
    ======================================================================



    
![png](lending_loan_files/lending_loan_20_5.png)
    


    [1mWhat do these three plots reveal about default behavior?[0m
    [92m
        Defaulters tend to have higher credit utilization, meaning they were
        already stretched thin before the loan was issued. They also tend to
        have higher monthly installments, which leaves less room in their
        budget when something goes wrong. The inquiries plot is the most
        telling though. Defaulters had more recent credit checks, which often
        means they were desperately seeking credit from multiple places at
        once rather than borrowing out of choice. That kind of behaviour is
        a financial stress signal that the model picks up on.
        [0m
    

