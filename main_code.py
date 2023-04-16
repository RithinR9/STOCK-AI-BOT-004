from tkinter import TRUE
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
import os
from datetime import datetime, timedelta

# Calculate the time range for the API URL (last 30 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
period1 = int(start_date.timestamp())
period2 = int(end_date.timestamp())
import requests
import urllib.request
from urllib.error import HTTPError

print("if you want to know stock symbols like AAPL(apple) got this link: https://business.unl.edu/outreach/econ-ed/nebraska-council-on-economic-education/student-programs/stock-market-game/documents/Top%202000%20Valued%20Companies%20with%20Ticker%20Symbols.pdf")
csv = input("Please Enter Which Stock You Would Like To Know About (ex. type AAPL(apple)): ")

# Fetch data from Yahoo Finance API
while True:
    # Fetch data from Yahoo Finance API
    print('Fetching data...')
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{csv}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    print(url)
    try:
        response = urllib.request.urlopen(url)
    except HTTPError as e:
        if e.code == 404:
            print("HTTP error 404: The requested page was not found on the server.")
        else:
            print(f"HTTP error {e.code}: {e.reason}")
    else:
        print("The page was found.")
    # Read the CSV file directly from the URL into a DataFrame
    response = requests.get(url)
    if response.status_code == 404:
        print('404 error: Page not found. PLEASE TYPE ACTUAL SYMBOL LIKE AAPL (apple)')
        csv = input("Please Enter Which Stock You Would Like To Know About (ex. type AAPL(apple)): ")
    else:
        print('Page found.')
        break

# Read the CSV file directly from the URL into a DataFrame
    

df = pd.read_csv(url)
print("loading csv from yahoo...")

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
while True:

    restart = input("Do you wish to try again y/n (it will exit if you type n): ")
    if(restart == "y"):
        python = sys.executable
        os.execl(python, python, *sys.argv)
        sys.exit()
    elif(restart == "n"):
        sys.exit()
    else:
        print("type either y or n")
