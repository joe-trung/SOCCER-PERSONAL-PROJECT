import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def load_fifa_data():
    dataset = {}
    for i in range(18, 24):
        dataset["Y20" + str(i)] = pd.read_csv('raw_data/FIFA' + str(i) + '_official_data.csv')
    return dataset


def keep_columns(dataset, keep_cols):
    for key in dataset:
        dataset[key] = dataset[key].loc[:, keep_cols]
    return dataset


# Load the FIFA data
fifa_data = load_fifa_data()

# Define the columns to keep
keep_cols = ['Name', 'Age', 'Overall', 'Potential', 'Club', 'Value', 'Wage', 'Special', 'Preferred Foot', 'Work Rate',
             'Height', 'Weight', 'Release Clause']

# Keep the specified columns from the FIFA data
fifa_data = keep_columns(fifa_data, keep_cols)

# Print the shape of each DataFrame to verify that the columns were dropped
for key in fifa_data:
    print(key, fifa_data[key].shape)


def plot_top_n_players(data, column, n=10):
    sort_df = data.sort_values(by=column, ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(x='Name', y=column, data=sort_df)

    # Create the Plot folder if it doesn't exist
    if not os.path.exists('Plot'):
        os.mkdir('Plot')

    # Save the plot as a PNG file in the Plot folder
    plt.savefig(f'Plot/{column}_barplot{n}.png')
    plt.close()


# Select the 2023 data
df23 = fifa_data['Y2023']
df22 = fifa_data['Y2022']
df21 = fifa_data['Y2021']
df20 = fifa_data['Y2020']
df19 = fifa_data['Y2019']
df18 = fifa_data['Y2018']

# Call the method to plot the top 10 players by Overall rating
plot_top_n_players(df23, 'Overall', 10)


def plot_boxplot(data, column):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.boxplot(x=data[column])

    # Save the plot as a PNG file in the Plot folder
    plt.savefig(f'Plot/{column}_boxplot.png')
    plt.close()


# Call the method to plot a box plot of the Overall column
plot_boxplot(df23, 'Overall')


def save_lineplot(data, x_col, y_col, output_filename):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.lineplot(data=data, x=x_col, y=y_col)

    # Save the plot as a PNG file in the Plot folder
    plt.savefig(f'Plot/{output_filename}.png')
    plt.close()


# Sort by Overall and select the top 100 players
sort100_df23 = df23.sort_values(by='Overall', ascending=False).head(100)

# Call the function to create the plot and save it to a file
save_lineplot(sort100_df23, 'Age', 'Overall', 'top100_age_overall')
save_lineplot(sort100_df23, 'Age', 'Value', 'top100_value_overall')


def convert_height_to_cm(height_str):
    # Split height string into feet and inches
    height_parts = height_str.split("'")
    feet = int(height_parts[0])
    inches = int(height_parts[1].replace('"', ''))

    # Compute height in cm
    total_inches = feet * 12 + inches
    height_cm = total_inches * 2.54

    return height_cm


def convert_weight_lb_to_kg(weight_lb):
    # Convert weight in pounds to kilograms
    weight_kg = int(int(weight_lb[:-3]) * 0.45359237)

    return weight_kg


def preprocess_dataframe(df):
    # Extract weight
    weight = []
    for i in df["Weight"]:
        if i[-2:] == 'kg':
            weight.append(int(i[:-2]))
        else:
            weight.append(convert_weight_lb_to_kg(i))

    # Extract height
    height = []
    for i in df["Height"]:
        if i[-2:] == 'cm':
            height.append(int(i[0:3]))
        else:
            height.append(convert_height_to_cm(i))

    # Compute BMI
    df['BMI'] = [round(w / (h / 100) ** 2, 2) for w, h in zip(weight, height)]

    return df


for df in (df18, df19, df20, df21, df22, df23):
    preprocess_dataframe(df)


def preprocess_fifa_data(data):
    merge = []
    for i in range(2018, 2024):
        data['Y' + str(i)]["Year"] = str(i)
        merge.append(data["Y" + str(i)])
    data = pd.concat(merge)
    # grouped_data = data.groupby(['Name']).mean()
    return data


preprocessed_data = preprocess_fifa_data(fifa_data)

# Save the processed data to a CSV file in the processed_data folder
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

preprocessed_data.to_csv('processed_data/fifa_data.csv')
