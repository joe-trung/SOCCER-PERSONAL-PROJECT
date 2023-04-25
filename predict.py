from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("my_app").config("spark.ui.port", "4050").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("INFO")


rdd = sc.textFile("s3://soccerpassionproject/processed_data/fifa_data.csv")
rdd.take(5)