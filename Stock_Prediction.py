# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Load the dataset
apple = pd.read_csv("aapl.csv")

# Initial data exploration
print(apple.head())
print("Training days =", apple.shape)

# Data visualization
sns.set()
plt.figure(figsize=(10, 4))
plt.title("Apple's Stock Price")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close"])
plt.show()

# Data preprocessing
apple = apple[["Close"]]
futureDays = 25
apple["Prediction"] = apple[["Close"]].shift(-futureDays)

# Prepare the features and target
x = np.array(apple.drop(["Prediction"], 1))[:-futureDays]
y = np.array(apple["Prediction"])[:-futureDays]

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

# Build and train the Decision Tree Regressor model
tree = DecisionTreeRegressor().fit(xtrain, ytrain)

# Build and train the Linear Regression model
linear = LinearRegression().fit(xtrain, ytrain)

# Prepare data for making future predictions
xfuture = apple.drop(["Prediction"], 1)[:-futureDays]
xfuture = xfuture.tail(futureDays)
xfuture = np.array(xfuture)

# Make predictions with the Decision Tree model
treePrediction = tree.predict(xfuture)
print("Decision Tree prediction =", treePrediction)

# Make predictions with the Linear Regression model
linearPrediction = linear.predict(xfuture)
print("Linear regression Prediction =", linearPrediction)

# Visualize the predictions from the Decision Tree model
predictions = treePrediction
valid = apple[x.shape[0]:]
valid["Predictions"] = predictions
plt.figure(figsize=(10, 6))
plt.title("Apple's Stock Price Prediction Model (Decision Tree Regressor Model)")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Original", "Valid", "Predictions"])
plt.show()

# Visualize the predictions from the Linear Regression model
predictions = linearPrediction
valid = apple[x.shape[0]:]
valid["Predictions"] = predictions
plt.figure(figsize=(10, 6))
plt.title("Apple's Stock Price Prediction Model (Linear Regression Model)")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Original", "Valid", "Predictions"])
plt.show()
