import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from openai import OpenAI
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Initialize sentiment analyzer and spaCy model
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Load credentials from environment variables
import os
reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
reddit_username = os.environ.get('REDDIT_USERNAME')
reddit_password = os.environ.get('REDDIT_PASSWORD')
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Initialize PRAW with credentials
reddit = praw.Reddit(
	client_id = reddit_client_id,
	client_secret = reddit_client_secret,
	username = reddit_username,
	password = reddit_password,
	user_agent = "clinical_trial_user_engagement"
)

# Define target subreddits
subreddits = ['health', 'clinicaltrials', 'AskDocs']

# Set up data storage
posts_data = []
comments_data = []

# Define functions

def fetch_and_analyze_posts():
	for subreddit in subreddits:
		for submission in reddit.subreddit(subreddit).new(limit=5): # Limit to 5 posts per subreddit, adjust as needed
			post_data = process_post(submission)
			posts_data.append(post_data)
			comments = submission.comments.list() if submission.num_comments > 0 else []
			for comment in comments:
				if isinstance(comment, praw.models.Comment):
					comment_data = process_comment(comment)
					comments_data.append(comment_data)

def process_post(post):
	text = post.title + " " + post.selftext
	sentiment = sia.polarity_scores(text)['compound']
	entities = extract_entities(text)
	sensitive_flag = detect_sensitive_content(text)
	return {
		"Type": "Post",
		"Content": text,
		"Sentiment Score": sentiment,
		"Entities": entities,
		"Sensitive Flag": sensitive_flag,
		"Created_UTC": datetime.utcfromtimestamp(post.created_utc),
		"Upvotes": post.score
	}

def process_comment(comment):
	text = comment.body
	sentiment = sia.polarity_scores(text)['compound']
	entities = extract_entities(text)
	sensitive_flag = detect_sensitive_content(text)
	return {
		"Type": "Comment",
		"Content": text,
		"Sentiment Score": sentiment,
		"Entities": entities,
		"Sensitive Flag": sensitive_flag,
		"Created_UTC": datetime.utcfromtimestamp(comment.created_utc)
	}

def extract_entities(text):
	doc = nlp(text)
	return [ent.text for ent in doc.ents]

def detect_sensitive_content(text):
	sensitive_keywords = ["diagnosis", "symptoms", "prescription", "treatment"]
	return any(re.search(r'\b' + kw + r'\b', text, re.IGNORECASE) for kw in sensitive_keywords)

def segment_user_by_interest(sentiment_score):
	if sentiment_score > 0.3:
		return "High"
	elif sentiment_score > 0:
		return "Medium"
	return "Low"

def generate_personalized_message(segment, content):
	
	client = OpenAI(
		api_key=openai_api_key,
	)
	
	prompt = (
		f"Generate a short personalized message aimed at a user who is in the '{segment}' "
		f"interest segment, based on the following content: '{content}'. "
		f"The message should encourage them to consider participating in a clinical trial, "
		f"highlighting potential benefits and addressing any concerns they may have."
	)
	completion = client.chat.completions.create(
		model="gpt-4o-mini", # Adjust as needed
		messages=[
			{
				"role": "system", 
				"content": "You are a helpful assistant."
			},
			{
				"role": "user",
				"content": prompt
			}
		],
		max_tokens=50 # Adjust as needed
	)
	generated_message = completion.choices[0].message.content.strip()
	return generated_message

def perform_clustering(data):
	tfidf_vectorizer = TfidfVectorizer(stop_words='english')
	tfidf_matrix = tfidf_vectorizer.fit_transform([d['Content'] for d in data])
	kmeans = KMeans(n_clusters=3, random_state=0)
	clusters = kmeans.fit_predict(tfidf_matrix)
	return clusters

def save_data_to_csv(data):
	df = pd.DataFrame(data)
	df.to_csv("clinical_trial_sentiment_data.csv", index=False)

# Run Analysis
fetch_and_analyze_posts()

# Segment and generate messages
all_data = posts_data + comments_data
for item in all_data:
	segment = segment_user_by_interest(item['Sentiment Score'])
	item['Segment'] = segment
	item['Message'] = generate_personalized_message(segment, item['Content'])

# Perform clustering
clusters = perform_clustering(all_data)
for idx, item in enumerate(all_data):
	item['Cluster'] = clusters[idx]

# Save data
save_data_to_csv(all_data)

# Plot Sentiment Distribution
def plot_sentiment_distribution(data):
	plt.figure(figsize=(10, 6))
	sns.histplot([d['Sentiment Score'] for d in data], bins=30, kde=True)
	plt.title("Sentiment Score Distribution")
	plt.xlabel("Sentiment Score")
	plt.ylabel("Frequency")
	plt.savefig("sentiment_distribution.png")
	plt.show()

# Plot Sentiment by Segment
def plot_sentiment_by_segment(data):
	df = pd.DataFrame(data)
	plt.figure(figsize=(10, 6))
	sns.boxplot(x='Segment', y='Sentiment Score', data=df)
	plt.title("Sentiment Scores by User Segment")
	plt.xlabel("User Segment")
	plt.ylabel("Sentiment Score")
	plt.savefig("sentiment_by_segment.png")
	plt.show()

# Plot Count of Posts and Comments by Sentiment Segment
def plot_count_by_segment(data):
	df = pd.DataFrame(data)
	segment_counts = df['Segment'].value_counts()
	plt.figure(figsize=(10, 6))
	sns.barplot(x=segment_counts.index, y=segment_counts.values)
	plt.title("Count of Posts and Comments by Sentiment Segment")
	plt.xlabel("User Segment")
	plt.ylabel("Count")
	plt.savefig("count_by_segment.png")
	plt.show()

# Plot Sentiment Score vs Upvotes
def plot_sentiment_vs_upvotes(data):
	df = pd.DataFrame(data)
	df = df[df['Type'] == 'Post']  # Only for posts since comments don't have upvotes
	plt.figure(figsize=(10, 6))
	sns.scatterplot(x='Upvotes', y='Sentiment Score', data=df, alpha=0.6)
	plt.title("Sentiment Score vs Upvotes")
	plt.xlabel("Number of Upvotes")
	plt.ylabel("Sentiment Score")
	plt.savefig("sentiment_vs_upvotes.png")
	plt.show()

# Plot Sentiment by Cluster
def plot_sentiment_by_cluster(data):
	df = pd.DataFrame(data)
	plt.figure(figsize=(10, 6))
	sns.boxplot(x='Cluster', y='Sentiment Score', data=df)
	plt.title("Sentiment Scores by Cluster")
	plt.xlabel("Cluster")
	plt.ylabel("Sentiment Score")
	plt.savefig("sentiment_by_cluster.png")
	plt.show()

# Plot Count of Posts and Comments by Cluster
def plot_count_by_cluster(data):
	df = pd.DataFrame(data)
	cluster_counts = df['Cluster'].value_counts()
	plt.figure(figsize=(10, 6))
	sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
	plt.title("Count of Posts and Comments by Cluster")
	plt.xlabel("Cluster")
	plt.ylabel("Count")
	plt.savefig("count_by_cluster.png")
	plt.show()

# Call plotting functions after data processing
plot_sentiment_distribution(all_data)
plot_sentiment_by_segment(all_data)
plot_count_by_segment(all_data)
plot_sentiment_vs_upvotes(all_data)

# New cluster visualizations
plot_sentiment_by_cluster(all_data)
plot_count_by_cluster(all_data)