from io import BytesIO
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# SAVE THESE INCASE NEED TO SAVE FILES LOCALLY
# from sklearn import preprocessing
# import os


from data_transform import fifa_data, preprocess_fifa_data


def plot_top_n_players(data, column, n=10):
    sort_df = data.sort_values(by=column, ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(x='Name', y=column, data=sort_df)

    # KEEP THESE CODE INCASE SWITCHING BACK TO LOCAL DRIVE
    # Create the Plot folder if it doesn't exist
    # if not os.path.exists('Plot'):
    #     os.mkdir('Plot')
    #
    # # Save the plot as a PNG file in the Plot folder
    # plt.savefig(f'Plot/{column}_barplot{n}.png')
    # plt.close()

    # Connect to S3
    s3 = boto3.resource('s3')

    # Save the plot as a PNG file in memory
    png_buffer = BytesIO()
    plt.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # Upload the plot to S3 as an object
    bucket_name = 'soccerpassionproject'
    object_key = f'plot/{column}_barplot{n}.png'
    s3.Bucket(bucket_name).put_object(Key=object_key, Body=png_buffer)
    s3.ObjectAcl(bucket_name, object_key).put(ACL='public-read')

    # Close the plot to free up memory
    plt.close()


# Not many years, do this better than loop
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
    # plt.savefig(f'Plot/{column}_boxplot.png')
    # plt.close()

    # Connect to S3
    s3 = boto3.resource('s3')

    # Save the plot as a PNG file in memory
    png_buffer = BytesIO()
    plt.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # Upload the plot to S3 as an object
    bucket_name = 'soccerpassionproject'
    object_key = f'plot/{column}_boxplot.png'
    s3.Bucket(bucket_name).put_object(Key=object_key, Body=png_buffer)
    s3.ObjectAcl(bucket_name, object_key).put(ACL='public-read')

    # Close the plot to free up memory
    plt.close()


def save_lineplot(data, x_col, y_col, output_filename):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.lineplot(data=data, x=x_col, y=y_col)

    # Save the plot as a PNG file in the Plot folder
    # plt.savefig(f'Plot/{output_filename}.png')
    # plt.close()

    # Connect to S3
    s3 = boto3.resource('s3')

    # Save the plot as a PNG file in memory
    png_buffer = BytesIO()
    plt.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # Upload the plot to S3 as an object
    bucket_name = 'soccerpassionproject'
    object_key = f'plot/{output_filename}.png'
    s3.Bucket(bucket_name).put_object(Key=object_key, Body=png_buffer)
    s3.ObjectAcl(bucket_name, object_key).put(ACL='public-read')

    # Close the plot to free up memory
    plt.close()

def plot_radar(dataset):
    labels = ['Overall', 'Special', 'Potential', 'Age', 'Value', 'Wage']
    # dataset = dataset[labels]
    numb = len(labels)
    # PLAYER BY NAME
    player_1 = 'Cristiano Ronaldo'
    values = list(dataset.loc[dataset['Name'] == player_1][labels])

    # CAN PUT DIFFERENT NAMEs TO RETRIEVE DIFFERENT RECORDS
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
    # fig.savefig("Plot/radar_plot.png")

    # Connect to S3
    s3 = boto3.resource('s3')

    # Save the plot as a PNG file in memory
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # Upload the plot to S3 as an object
    bucket_name = 'soccerpassionproject'
    object_key = f'plot/radar_plot.png'
    s3.Bucket(bucket_name).put_object(Key=object_key, Body=png_buffer)
    s3.ObjectAcl(bucket_name, object_key).put(ACL='public-read')

    # Close the plot to free up memory
    plt.close()


if __name__ == '__main__':
    # Top 10 players by Overall rating
    plot_top_n_players(df23, 'Overall', 10)

    # Box plot of the Overall column
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

