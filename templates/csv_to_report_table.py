import boto3
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO


s3 = boto3.client('s3')


bucket_name = 'soccerpassionproject'
csv_file_name = 'ml_models/r_square_score.csv'
png_file_name = 'ml_models/table.png'

# read the CSV file from S3 into a pandas dataframe
obj = s3.get_object(Bucket=bucket_name, Key=csv_file_name)
df = pd.read_csv(obj['Body'])

# create a table plot using pandas and matplotlib
fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# save the plot as a PNG file to a bytes buffer
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)

# upload the PNG file to S3
s3.put_object(Bucket=bucket_name, Key=png_file_name, Body=buffer)
s3.put_object_acl(Bucket=bucket_name, Key=png_file_name, ACL='public-read')