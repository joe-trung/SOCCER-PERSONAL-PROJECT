import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import os
import numpy as np

from data_transform import fifa_data, preprocess_fifa_data


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


def plot_boxplot(data, column):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.boxplot(x=data[column])

    # Save the plot as a PNG file in the Plot folder
    plt.savefig(f'Plot/{column}_boxplot.png')
    plt.close()


def save_lineplot(data, x_col, y_col, output_filename):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.lineplot(data=data, x=x_col, y=y_col)

    # Save the plot as a PNG file in the Plot folder
    plt.savefig(f'Plot/{output_filename}.png')
    plt.close()

def plot_radar(dataset):
    labels = ['Overall', 'Special', 'Potential', 'Age', 'Value', 'Wage']
    # dataset = dataset[labels]
    numb = len(labels)
    # PLAYER BY NAME
    player_1 = 'Cristiano Ronaldo'
    values = list(dataset.loc[dataset['Name'] == player_1][labels])
    # player_2 = 'L. Messi'
    # values = dataset.loc[dataset['Name'] == player_2][labels])
    # PLAYER BY LOCATION
    # values = list(dataset.loc[1000])

    angles = np.linspace(0, 2 * np.pi, numb, endpoint=False).tolist()

    # Plot is a circle, we need the loop and append the start to the end
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw outline for data
    ax.plot(angles, values, color='blue', linewidth=1)

    ax.fill(angles, values, color='red', alpha=0.25)
    ax.set_title('Radar Plot')

    # Save plot to file
    fig.savefig("Plot/radar_plot.png")


if __name__ == '__main__':
    # Call the method to plot the top 10 players by Overall rating
    plot_top_n_players(df23, 'Overall', 10)

    # Call the method to plot a box plot of the Overall column
    plot_boxplot(df23, 'Overall')

    # Sort by Overall and select the top 50 players
    sort50_df23 = df23.sort_values(by='Overall', ascending=False).head(50)

    # Call the function to create the plot and save it to a file
    save_lineplot(sort50_df23, 'Age', 'Overall', 'top50_age_overall')
    save_lineplot(sort50_df23, 'Value', 'Age', 'top50_value_age')
    save_lineplot(sort50_df23,  'Value', 'Overall', 'top50_value_overall')
    save_lineplot(sort50_df23,  'Value', 'Wage', 'top50_value_wage')

    # RADAR PLOT
    plot_radar(preprocess_fifa_data(fifa_data))

    print('Successfully exported visualized files in to Plot folder')

