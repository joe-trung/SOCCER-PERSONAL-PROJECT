import boto3
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Get table
table = dynamodb.Table('your_table_name')

# Get all comments
response = table.scan()
comments = [item['comment'] for item in response['Items']]

# Define stop words to exclude
stop_words = set(stopwords.words('english'))

# Define punctuation to exclude
punctuation = set(['.', ',', ';', ':', '?', '!', '-', '(', ')'])

# Function to clean and tokenize text
def clean_text(text):
    # Convert to lower case
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words and punctuation
    tokens = [token for token in tokens if not token in stop_words and not token in punctuation]
    # Remove any non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

# Clean and tokenize all comments
all_tokens = []
for comment in comments:
    tokens = clean_text(comment)
    all_tokens.extend(tokens)

# Get the most common words
word_counts = Counter(all_tokens)
top_words = word_counts.most_common(10)

# Plot the most common words
labels, values = zip(*top_words)
plt.bar(labels, values)
plt.title('Most Common Words')
plt.xlabel('Word')
plt.ylabel('Count')
plt.show()
