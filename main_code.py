import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys

# Fetch data from Yahoo Finance API
print('Fetching data...')
url = "https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1649050272&period2=1680586272&interval=1d&events=history&includeAdjustedClose=true"
response = requests.get(url)

# Read the CSV file directly from the URL into a DataFrame
df = pd.read_csv(url)
print("loading aapl-csv from yahoo...")

# Add a column to the DataFrame that predicts the percentage change in stock prices from one day to the next
df['Daily % Change'] = (df['Close'].pct_change() * 100).round(2)

# Shift the column to create the prediction target for the next day
df['Next Day % Change'] = df['Daily % Change'].shift(-1)

# Drop the last row since it doesn't have a prediction target
df.drop(df.tail(1).index, inplace=True)

# Split the data into training and testing sets
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Next Day % Change']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a linear regression model
print('Training model...')
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the testing set
print('Making predictions...')
y_pred = reg.predict(X_test)

# Show the predicted percentage change for the next day
print('Predicted percentage change for the next day:')
print()
print(y_pred[-1])
print()

# Add a column to the DataFrame that predicts whether the stock will show a profit or a loss
df['Profit/Loss'] = df['Close'].diff().apply(lambda x: 'Profit' if x > 0 else 'Loss')

# Shift the column to create the prediction target for the next day
df['Next Day Profit/Loss'] = df['Profit/Loss'].shift(-1)

# Drop the last row since it doesn't have a prediction target
df.drop(df.tail(1).index, inplace=True)

print('predicting...')

# Split the data into training and testing sets
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Next Day Profit/Loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate accuracy as a percentage
accuracy = accuracy_score(y_test, y_pred) * 100
print('Predicted percentage of profit and loss for the next day:')
print()
print(f"{accuracy:.2f}%")
print()
if accuracy > 50:
    print('higher rate of profit')
    print()
else:
    print('higher rate of loss')
    print()
print()
print("(DON'T ACTUALLY USE THIS FOR STOCK MARKETING BUT THIS IS A DEMONSTRATION OF A STOCK MARKET BOT)")
print("A PROJECT BY RITHIN.R, PLEASE HELP ME GET TO TOP 3")
print()
input("Press Enter to EXIT...")
sys.exit()
