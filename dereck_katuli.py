import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Below code will read the bank mock dataset.csv file. We will save it to a dataframe called df.
df = pd.read_csv('./bank-mock-dataset.csv')


# A bit of exploratory of the dataset
print(df.head(2))
print("\n")
print(df.shape) # This data has 1000 rows and 2 columns
print("\n")
print(df.info())
print("\n")
print(df.describe())
print("\n")


# Below we will get the different features from the dataset and assign to X variable
X = df[['age', 'salary','credit_history_depth', 'loan_amount','interest_income', 'fees_income', 'operational_cost','risk_score',
        'transaction_volume','digital_channel_usage','monthly_profit']]

target = df['default_status']


# Let's split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    target, 
    test_size=0.2,
    random_state=42)

# Let's get the shape for the training dataset
# print(X_train.shape)

# Let's get the scaled features using RobustScaler. Here we are standardizing the features
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Let's cluster our data into 2 clusters using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

y_pred = kmeans.predict(X_scaled)


# ==========================================
# Section B: Supervised Learning (Profit Forecasting Model)
# ==========================================

# Let's create a new variable predicted_profit_next_year as our target variable
df['predicted_profit_next_year'] = df['monthly_profit'] * 1.05

#Train at least two supervised ML models using Python:

# 1. Let's train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 2. Let's train the model using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the model
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)

# Compare model performance using R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

# Getting a glimpse of 5 records from the y_train dataset.
# print(y_train.head(5))


clusters = kmeans.predict(X_scaled)
labels = kmeans.fit_predict(X_scaled)

u_labels = np.unique(labels)
for i in u_labels:
    plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f'Cluster {i}')

# Step 4: Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k', marker='X', label='Centroids')

plt.legend()
plt.title("K-Means Clustering Visualization")
plt.show()

# Scenario A : Apply a 10–15% increase in revenue for high-value clusters. 
increase_percentage = 0.15
high_value_cluster = 2
df.loc[clusters == high_value_cluster, 'predicted_profit_next_year'] *= (1 + increase_percentage)
print(df['predicted_profit_next_year'])

# Scenario B: Apply a 20% cost reduction for low-value clusters.
reduction_percentage = 0.20
low_value_cluster = 0
df.loc[clusters == low_value_cluster, 'operational_cost'] *= (1 - reduction_percentage)

# plt.plot()


# Scenario C: Reduce lending exposure for high-risk clusters by:
    # •	lowering loan amounts



    # •	reducing interest-based income expectations

# Part D — Executive Summary
# Write a short summary in md file (max 300 words) addressing:
# 1.	How supervised and unsupervised learning complemented each other.
"""In supervised learning we have labels while unsupervised learning we do not have any labels.    """

# 2.	Which strategy the bank should adopt for maximum profit growth.


# 3.	Which risk areas the bank must monitor if these strategies go live.
"""

"""

# 4.	A recommended monitoring framework using ML outputs.

# A chart for Baseline forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['monthly_profit'], label='Baseline Monthly Profit', color='blue')
plt.plot(df.index, df['predicted_profit_next_year'], label='Predicted Profit Next   Year', color='orange')
plt.xlabel('Index')
plt.ylabel('Profit')
plt.title('Baseline vs Predicted Profit Next Year')
plt.legend()
plt.show()




