import boto3
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Set up DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Get table
table = dynamodb.Table('comments')

# Get all comments
response = table.scan()
comments = [item['comment'] for item in response['Items']]

# Define stop words to exclude
stop_words = set(stopwords.words('english'))

# Define punctuation to exclude
others = set(['.', ',', ';', ':', '?', '!', '-', '(', ')', 'project', 'comment', 'work', 'see', 'put'])


# Function to clean and tokenize text
def clean_text(text):
    # Convert to lower case
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words and punctuation
    tokens = [token for token in tokens if not token in stop_words and not token in others]
    # Remove any non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]
    return tokens


# Clean and tokenize all comments
all_tokens = []
for comment in comments:
    tokens = clean_text(comment)
    all_tokens.extend(tokens)

# Create word cloud
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate_from_frequencies(
    Counter(all_tokens))

# Display the generated image:
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("Plot/feeback.png")
