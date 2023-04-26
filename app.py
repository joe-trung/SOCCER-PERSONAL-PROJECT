import pandas as pd
from flask import Flask, render_template, request, flash
import boto3
import time

app = Flask(__name__)
app.secret_key = 'TRUNG TRAN'


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/visualize')
def visualize():
    return render_template('visualize.html')


@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/ml')
def ml():
    return render_template('ml.html')


@app.route('/comment', methods=['GET'])
def comment():
    return render_template('comment.html')


@app.route('/comment', methods=['POST'])
def add_comment():
    name = request.form['name']
    comment = request.form['comment']

    # Add comment to DynamoDB table
    table.put_item(
        Item={
            'comment_id': str(time.time()),
            'name': name,
            'comment': comment
        }
    )
    flash('Thank you for your comment! It was saved on DynamoDB')
    return render_template('comment.html')


# COMMAND IN CLI TO DOWNLOAD FROM DYNAMODB
# aws dynamodb scan --table-name comments --output json --query "Items[*]" > json/comments.json


dynamodb = boto3.resource('dynamodb')

table_name = 'comments'

# Check if table already exists
existing_tables = list(dynamodb.tables.all())
if any(table.name == table_name for table in existing_tables):
    table = dynamodb.Table(table_name)
else:
    # Create comments table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'comment_id',
                'KeyType': 'HASH'  # Partition key
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'comment_id',
                'AttributeType': 'S'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )

    # Wait for table to be created
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)


# THIS CODE FOR THE GAME
# Load player data from 2023 CSV file
player_data = pd.read_csv("raw_data/FIFA23_official_data.csv")


@app.route("/game", methods=["GET", "POST"])
def game():
    player_name = None
    player_info_list = []

    if request.method == "POST":
        player_name = request.form["player_name"]

        # Retrieve player information from player_data DataFrame
        player_info_df = player_data.loc[player_data["Name"].str.lower().str.contains(player_name.lower())]

        if not player_info_df.empty:
            for _, row in player_info_df.iterrows():
                player_info = {
                    "name": row["Name"],
                    "club": row["Club"],
                    "age": row["Age"],
                    "value": row["Value"]
                }
                player_info_list.append(player_info)

    return render_template("game.html", player_name=player_name, player_info_list=player_info_list)


if __name__ == "__main__":
    app.run(debug=True)
