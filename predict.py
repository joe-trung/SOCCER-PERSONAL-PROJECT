from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer

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

def heat_map(df):



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

