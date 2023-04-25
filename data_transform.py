import os

import boto3
import pandas as pd


def load_fifa_data():
    dataset = {}
    for i in range(18, 24):
        # dataset["Y20" + str(i)] = pd.read_csv('raw_data/FIFA' + str(i) + '_official_data.csv')
        dataset["Y20" + str(i)] = pd.read_csv('https://soccerpassionproject.s3.amazonaws.com/raw_data/FIFA'+str(i)+'_official_data.csv')
    return dataset


# Load the FIFA data
fifa_data = load_fifa_data()
# Select and process every year's data
df23 = fifa_data['Y2023']
df22 = fifa_data['Y2022']
df21 = fifa_data['Y2021']
df20 = fifa_data['Y2020']
df19 = fifa_data['Y2019']
df18 = fifa_data['Y2018']


def keep_columns(dataset, keep_cols):
    for key in dataset:
        dataset[key] = dataset[key].loc[:, keep_cols]
    return dataset


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


def preprocess_fifa_data(data):
    merge = []
    for i in range(2018, 2024):
        data['Y' + str(i)]["Year"] = str(i)
        merge.append(data["Y" + str(i)])
    data = pd.concat(merge)
    # grouped_data = data.groupby(['Name']).mean()
    return data


if __name__ == "__main__":
    # Define the columns to keep
    keep_cols = ['Name', 'Age', 'Overall', 'Potential', 'Club', 'Value', 'Wage', 'Special', 'Preferred Foot',
                 'Work Rate', 'Height', 'Weight', 'Release Clause']

    # Keep the specified columns from the FIFA data
    fifa_data = keep_columns(fifa_data, keep_cols)

    # Print the shape of each DataFrame to verify that the columns were dropped
    for key in fifa_data:
        print(key, fifa_data[key].shape)

    for df in (df18, df19, df20, df21, df22, df23):
        preprocess_dataframe(df)

    # Merge all processed dataframe into one
    preprocessed_data = preprocess_fifa_data(fifa_data)

    # Save the processed data to a CSV file in the processed_data folder
    # if not os.path.exists('processed_data'):
    #     os.makedirs('processed_data')
    # preprocessed_data.to_csv('processed_data/fifa_data.csv')

    # Save the processed data to an S3 bucket
    s3 = boto3.resource('s3')
    bucket_name = 'soccerpassionproject'
    # Convert the DataFrame to CSV format
    csv_buffer = preprocessed_data.to_csv(index=False).encode()
    # Upload the CSV data to S3
    s3.Object(bucket_name, 'processed_data/fifa_data.csv').put(Body=csv_buffer)
    s3.ObjectAcl(bucket_name, 'processed_data/fifa_data.csv').put(ACL='public-read')

    # Print something to confirm Success
    print("Exported final dataset with shape of:" + str(preprocessed_data.shape))
