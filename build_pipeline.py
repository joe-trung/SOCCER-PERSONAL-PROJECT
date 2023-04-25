import json
import http.client
import os



# THIS WAS BUILT TO COLLECT DATA FROM API, BUT LIMITED REQUEST EVERYDAY MAKE IT IMPOSSIBLE.
#  DATASET WAS COLLECTED FROM KAGGLE, FIFA...
def get_soccer_data(api_key):
    conn = http.client.HTTPSConnection("v3.football.api-sports.io")

    headers = {
        'x-rapidapi-host': "v3.football.api-sports.io",
        'x-rapidapi-key': api_key
    }

    conn.request("GET", "/players/seasons", headers=headers)

    response = conn.getresponse()
    data = response.read()

    # Decode the bytes object to a string
    data_str = data.decode('utf-8')

    # Parse the JSON string into a Python object
    players = json.loads(data_str)["response"]

    # Save the fixtures data to a JSON file
    with open("soccer_players.json", "w") as f:
        json.dump(players, f)


if __name__ == '__main__':
    # THIS KEY IS NOT IMPORTANT
    # api_key = os.environ.get('API_KEY')
    api_key = "9b86b389ee9d0cf5104df2e31e703a22"
    get_soccer_data(api_key)
