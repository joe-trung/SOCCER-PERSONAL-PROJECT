from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer
import seaborn as sns
import pandas as pd
import boto3
from io import BytesIO

spark = SparkSession.builder.appName('PassionProject').getOrCreate()
df = spark.read.csv('processed_data/fifa_data.csv', header=True, inferSchema=True)


def find_missing_values_columns(df):
    nrow = df.count()
    for v in df.columns:
        summary_df = df.select(v).describe()
        v_count = int(summary_df.collect()[0][v])
        if v_count < nrow:
            missing_rows = nrow - v_count
            missing_percentage = (1 - v_count / nrow) * 100
            print("Total observations: " + str(nrow) + "\n"
                    "Total observations of " + v + ": " + str(v_count) + "\n"
                    "Total missing rows is: " + str(missing_rows) + "\n"
                    "Percentage of missing values: " + str(missing_percentage) + "%" + "\n"
                    "----------------------------")


def make_currency_numeric(dataf):
    dataf = dataf.withColumn('Value in Mil', f.regexp_replace('Value', '[€,M]', '').cast('double'))
    dataf = dataf.withColumn('Wage in K', f.regexp_replace('Wage', '[€,K]', '').cast('double'))
    dataf = dataf.withColumn('Release Clause in Mil', f.regexp_replace('Release Clause', '[€,M]', '').cast('double'))
    return dataf


def indexing_string(df, column):
    # build indexer
    string_indexer = StringIndexer(inputCol=column, outputCol=f'{column}_')
    # learn the model
    string_indexer_model = string_indexer.fit(df)
    # transform the data
    df = string_indexer_model.transform(df)
    # resulting df
    return df


def fill_in_null(dataf):
    dataf = dataf.fillna({'Club': 'None'})
    dataf = dataf.fillna({'Release Clause in Mil': 0})
    return dataf


# Convert PySpark DataFrame to Pandas DataFrame
def convert_spark_df_to_pandas_df(spark_df):
    return spark_df.toPandas()


# Create heatmap correlation plot and save to S3
def create_heatmap_plot(panda_df):
    sns.set(style="white")
    corr = panda_df.corr()
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm")

    # Connect to S3
    s3 = boto3.resource('s3')

    # Save the plot as a PNG file in memory
    png_buffer = BytesIO()
    plt.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # Upload the plot to S3 as an object
    bucket_name = 'soccerpassionproject'
    object_key = f'plot/heatmap.png'
    s3.Bucket(bucket_name).put_object(Key=object_key, Body=png_buffer)
    s3.ObjectAcl(bucket_name, object_key).put(ACL='public-read')

    # Close the plot to free up memory
    plt.close()
    print('Exported heatmap')


def create_correlation_scatter(panda_df, column1, column2):
    # Load data and create scatter plot
    x_col = column1
    y_col = column2
    fig, ax = plt.subplots()
    ax.scatter(panda_df[x_col], panda_df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    # Connect to S3
    s3 = boto3.resource('s3')

    # Save the plot as a PNG file in memory
    png_buffer = BytesIO()
    plt.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # Upload the plot to S3 as an object
    bucket_name = 'soccerpassionproject'
    object_key = f'plot/corelation_{column1}_{column2}.png'
    s3.Bucket(bucket_name).put_object(Key=object_key, Body=png_buffer)
    s3.ObjectAcl(bucket_name, object_key).put(ACL='public-read')

    # Close the plot to free up memory
    plt.close()
    print('Exported correlation scatter')


if __name__ == '__main__':
    # Find missing values
    find_missing_values_columns(df)

    # Change currency values to float
    df1 = make_currency_numeric(df)

    # Make indexing for preferred foot.
    df2 = indexing_string(df1, "Preferred Foot")

    # Fill in null for Release Clause and Club
    df3 = fill_in_null(df2)

    # Select columns
    df4 = df3.select('Age', 'Overall', 'Potential', 'Special', 'BMI', 'Year',
                     'Wage in K', 'Release Clause in Mil', 'Preferred Foot_', 'Value in Mil')

    pdf = convert_spark_df_to_pandas_df(df4)
    create_heatmap_plot(pdf)
    create_correlation_scatter(pdf, 'Value in Mil', 'Age')
    create_correlation_scatter(pdf, 'Value in Mil', 'Overall')
    create_correlation_scatter(pdf, 'Value in Mil', 'Potential')
    create_correlation_scatter(pdf, 'Value in Mil', 'Special')
    create_correlation_scatter(pdf, 'Value in Mil', 'BMI')
    create_correlation_scatter(pdf, 'Value in Mil', 'Wage in K')
    create_correlation_scatter(pdf, 'Value in Mil', 'Release Clause in Mil')


