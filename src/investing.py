from training import training_main
import pandas as pd
import matplotlib.pyplot as plt

test_csv = "C:/Users/mario/Downloads/testing_dataset.csv"
winnings_csv = "C:/Users/mario/Downloads/Winnings.csv"

test_predictions = training_main()

df = pd.read_csv(winnings_csv)

selected_df = df[['Open', 'pct_change']].copy()
selected_df['Winnings'] = df['Open'].diff()
selected_df.loc[selected_df['pct_change'] == 0, 'Winnings'] = 0
selected_df['Predictions'] = test_predictions

predictions = selected_df['Predictions']
winnings = selected_df['Winnings']
change = selected_df['pct_change']
open_price = selected_df['Open']

profit = 0
profit_list = []

profit_1 = 0
profit_list_1 = []

profit_2 = 0
profit_list_2 = []

for i in range(len(predictions)):
    if predictions[i] == 0:
        profit_list_1.append(profit_1)
        profit_list_2.append(profit_2)
        profit += 100 / open_price[i] * winnings[i]
        profit_list.append(profit)

    else:
        profit_list.append(profit)

        if winnings[i] < 0:
            profit_list_2.append(profit_2)

        profit_1 += 100 / open_price[i] * winnings[i]
        profit_list_1.append(profit_1)

    if predictions[i] == 1 and winnings[i] > 0:
        profit_2 += 100 / open_price[i] * winnings[i]
        profit_list_2.append(profit_2)

selected_df.to_csv(winnings_csv, index=False)

plt.figure(figsize=(12,6))

'''for 100$ every day'''
plt.plot(profit_list, label='Profits when the model says to go long')
plt.plot(profit_list_1, label='Profits when the model says to short')
plt.plot(profit_list_2, label='Missing profits due to false negatives')

print(profit)
print(profit_1)
print(profit_2)

plt.xlabel('Date')
plt.ylabel('Profits')
plt.title('Model profits')
plt.legend()
plt.show()

print("Predictions have been saved")
