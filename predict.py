from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('PassionProject').getOrCreate()
df = spark.read.csv('processed_data/fifa_data.csv', header = True, inferSchema=True)

def find_missing_values_columns(df):
    nrow = df.count()
    for v in df.columns:
        summary_df = df.select(v).describe()
        v_count = int(summary_df.collect()[0][v])
        if v_count < nrow:
            missing_rows = nrow - v_count
            missing_percentage = (1 - v_count/nrow) * 100
            print("Total observations: " + str(nrow) + "\n"
                 "Total observations of " + v + ": " + str(v_count) + "\n"
                 "Total missing rows is: "+str(missing_rows)+"\n"
                 "Percentage of missing values: " + str(missing_percentage) + "%" + "\n"
                 "----------------------------")



if __name__ == '__main__':
    find_missing_values_columns(df)
