from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer, VectorAssembler
import seaborn as sns
import boto3
import pandas as pd
from io import BytesIO, StringIO
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

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


def make_currency_numeric(df):
    df1 = df.withColumn('Value in Mil', f.regexp_replace('Value', '[€,M]', '').cast('double'))
    df2 = df1.withColumn('Wage in K', f.regexp_replace('Wage', '[€,K]', '').cast('double'))
    df3 = df2.withColumn('Release Clause in Mil', f.regexp_replace('Release Clause', '[€,M]', '').cast('double'))
    return df3


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


def big_histogram(df):
    # Get number of rows and columns in DataFrame
    n_rows, n_cols = df.shape

    # Determine number of subplot rows and columns
    n_subplot_cols = 4
    n_subplot_rows = (n_cols + n_subplot_cols - 1) // n_subplot_cols

    # Create figure and axes
    fig, axs = plt.subplots(n_subplot_rows, n_subplot_cols, figsize=(20, 10))

    # Plot histograms for each column
    for i, col_name in enumerate(df.columns):
        row_idx = i // n_subplot_cols
        col_idx = i % n_subplot_cols
        ax = axs[row_idx, col_idx]
        ax.hist(df[col_name], bins=20)
        ax.set_title(col_name)

    # Adjust spacing between subplots
    fig.tight_layout()

    # Connect to S3
    s3 = boto3.resource('s3')

    # Save the plot as a PNG file in memory
    png_buffer = BytesIO()
    plt.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # Upload the plot to S3 as an object
    bucket_name = 'soccerpassionproject'
    object_key = f'plot/bighistogram.png'
    s3.Bucket(bucket_name).put_object(Key=object_key, Body=png_buffer)
    s3.ObjectAcl(bucket_name, object_key).put(ACL='public-read')

    # Close the plot to free up memory
    plt.close()
    print('Exported big histogram')


def vector_assembler(dataf, input_cols, output_col):
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    output_df = assembler.transform(dataf)
    return output_df


def train_linear_regression_model(df):
    # Split the input DataFrame into training and testing sets
    (trainingData, testData) = df.randomSplit([0.8, 0.2])

    # Create a LinearRegression object with the desired parameters
    lr = LinearRegression(maxIter=10)

    # Train the model on the training data
    lrModel = lr.fit(trainingData)

    # Use the model to make predictions on the test data
    predictions = lrModel.transform(testData)

    # Evaluate the model using the R2 metric
    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="label", metricName="r2")
    r2 = evaluator.evaluate(predictions)

    # Return the trained model and R2 score
    return r2


def decision_tree_regression(df):
    # Define the feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=["features"], outputCol="features_vec")
    data = assembler.transform(df)

    # Split the data into training and testing sets
    (train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

    # Define the Decision Tree model
    dt = DecisionTreeRegressor(featuresCol="features_vec", labelCol="label")

    # Train the model using the training data
    model = dt.fit(train_data)

    # Make predictions using the testing data
    predictions = model.transform(test_data)

    # Evaluate the model using R-squared
    evaluator = RegressionEvaluator(metricName="r2")
    r2 = evaluator.evaluate(predictions)

    return r2


def random_forest_regression(df):
    # Define the feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=["features"], outputCol="features_vec")
    data = assembler.transform(df)

    # Split the data into training and testing sets
    (train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

    # Define the Random Forest model
    rf = RandomForestRegressor(featuresCol="features_vec", labelCol="label", numTrees=100, seed=42)

    # Train the model using the training data
    model = rf.fit(train_data)

    # Make predictions using the testing data
    predictions = model.transform(test_data)

    # Evaluate the model using R-squared
    evaluator = RegressionEvaluator(metricName="r2")
    r2 = evaluator.evaluate(predictions)

    return r2


def save_data_to_s3(r2, r3, r4):
    # Create a Pandas DataFrame to hold the R-squared scores
    data = {'Model': ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression'],
            'R-squared Score': [r2, r3, r4]}
    df = pd.DataFrame(data)

    # Convert the Pandas DataFrame to a CSV string
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Upload the CSV string to S3
    bucket_name = 'soccerpassionproject'
    key = 'ml_models/r_square_score.csv'
    s3.put_object(Bucket=bucket_name, Key=key, Body=csv_string,ACL='public-read')


if __name__ == '__main__':
    # Find missing values
    find_missing_values_columns(df)

    # Change currency values to float
    df = make_currency_numeric(df)

    # Make indexing for preferred foot.
    df = indexing_string(df, "Preferred Foot")

    # Fill in null for Release Clause and Club
    df = fill_in_null(df)
    df = df.na.drop(how='any')

    # Select columns
    df = df.select('Age', 'Overall', 'Potential', 'Special', 'BMI', 'Year',
                     'Wage in K', 'Release Clause in Mil', 'Preferred Foot_', 'Value in Mil')

    # Make more heatmap, histogram, and scatter plots
    pdf = convert_spark_df_to_pandas_df(df)
    create_heatmap_plot(pdf)
    create_correlation_scatter(pdf, 'Value in Mil', 'Age')
    create_correlation_scatter(pdf, 'Value in Mil', 'Overall')
    create_correlation_scatter(pdf, 'Value in Mil', 'Potential')
    create_correlation_scatter(pdf, 'Value in Mil', 'Special')
    create_correlation_scatter(pdf, 'Value in Mil', 'BMI')
    create_correlation_scatter(pdf, 'Value in Mil', 'Wage in K')
    create_correlation_scatter(pdf, 'Value in Mil', 'Release Clause in Mil')
    big_histogram(pdf)


    # Vector assembler
    input_cols = ['Age', 'Overall', 'Potential', 'Special', 'BMI', 'Year',
                  'Wage in K', 'Release Clause in Mil', 'Preferred Foot_']
    output_col = 'features'
    feature_df = vector_assembler(df, input_cols, output_col)
    new_df = feature_df.select('features', "Value in Mil").withColumnRenamed("Value in Mil", 'label')

    r2_linear = train_linear_regression_model(new_df)
    print("R2 Score from linear regression: ", r2_linear)

    r2_decision = decision_tree_regression(new_df)
    print("R2 Score from decision tree regression: ", r2_decision)

    r2_forest = random_forest_regression(new_df)
    print("R2 Score from random forest regression: ", r2_forest)

    s3 = boto3.client('s3')
    save_data_to_s3(r2_linear, r2_decision, r2_forest)
