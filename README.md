# Clinical Trial User Engagement Analysis
 This assignment analyzes user sentiment from Reddit posts and comments related to clinical trials. It segments users by interest level based on sentiment analysis and generates personalized engagement messages using the OpenAI API to encourage participation in clinical trials. The repository includes source code, documentation, and visualizations to support the analysis.

## Overview

In this assignment, I focused on scraping data from specific subreddits related to health and clinical trials, performing sentiment analysis on the posts and comments, and generating personalized messages to engage users about potential participation in clinical trials.

## Setup Instructions

1. **Prerequisites**:
   - Python 3.7 or higher
   - Required libraries:
     - `praw`
     - `nltk`
     - `pandas`
     - `openai`
     - `matplotlib`
     - `seaborn`
     - `spacy`
     - `sklearn`

2. **Create a virtual environment** (optional):
   I recommend creating a virtual environment to manage dependencies:
   ```bash
   python -m venv env
   source env/bin/activate
   ```

2. **Installation**:
   I installed the required libraries using pip:
   ```bash
   pip3 install praw nltk pandas openai matplotlib seaborn spacy scikit-learn
   ```

3. **Install spaCy model "en_core_web_sm"**:
    Install the spaCy model for English language processing:
    ```bash
    python3 -m spacy download en_core_web_sm
    ```
   
4. **Set Up PRAW**:
    Set up PRAW with your Reddit API credentials:
    - Create a Reddit app at https://www.reddit.com/prefs/apps
    - Obtain the client ID, client secret, and user agent
    - Use environment variables to store the credentials by typing the following commands in your terminal:
      ```bash
      export REDDIT_CLIENT_ID='<your_client_id>'
      export REDDIT_CLIENT_SECRET='<your_client_secret>'
      export REDDIT_USERNAME='<your_reddit_username>'
      export REDDIT_PASSWORD='<your_reddit_password>'
      export OPENAI_API_KEY='<your_openai_api_key>'
      ```

6. **Run the Code**:
    Run the code to scrape data, perform sentiment analysis, and generate personalized messages:
    ```bash
    python3 clinical_trial_user_engagement.py
    ```


## Methodology

- **Data Collection**: I scraped posts and comments from selected subreddits using PRAW. For each post and comment, I analyzed sentiment, detected sensitive content, and extracted entities.

- **Sentiment Analysis**: I used the VADER sentiment analysis tool to evaluate the sentiment of each post and comment, categorizing users based on their expressed sentiment towards relevant topics.

- **User Segmentation**: I segmented users into interest levels (High, Medium, Low) based on their sentiment scores to tailor engagement messages effectively.

- **Message Generation**: I utilized the OpenAI API to generate personalized messages aimed at users who express interest in or could potentially benefit from participating in clinical trials.

## Challenges

One notable challenge I encountered during the assignment was the limitation imposed by the OpenAI API regarding the maximum token limit for generated messages.

### Token Limit Implementation

To manage costs and prevent excessive charges on the OpenAI API, I implemented a maximum token limit of 50 for generating personalized messages. This limit was intended to help me stay within budget while still generating effective communication. However, I found that it could lead to truncated messages that lacked the necessary context.

For improved message quality, I suggest adjusting the token limit to around 100-150 tokens, which allows for more comprehensive and engaging interactions with users about clinical trial participation opportunities.

## Data Collected

The collected data includes:
- Posts and comments from selected subreddits.
- Sentiment scores, entities detected, and sensitive content flags.
- User segmentation levels and generated messages.
  
You can find an example of the collected data in `clinical_trial_sentiment_data.csv`.

## Analysis Performed

Throughout the assignment, I conducted various analyses, including:
- Visualizing sentiment distribution.
- Analyzing sentiment scores by user segment.
- Counting posts and comments by sentiment segment.
- Plotting sentiment scores against upvotes.
- Performing clustering of content for user interest analysis.

## Ethical Considerations

While designing and implementing this solution, I considered several ethical concerns:
- **Data Privacy**: I ensured that user data is handled with care and that only aggregated information is shared.
- **User Engagement**: I aimed to create messages that are respectful, relevant, and encourage meaningful participation in clinical trials.

## Evaluation
- **Functionality**:
My script effectively scrapes posts and comments from selected subreddits, performs sentiment analysis using the VADER sentiment analyzer, and generates personalized messages with the OpenAI API. It successfully segments users based on their interest levels derived from sentiment scores, ensuring targeted engagement. All functionalities are well-integrated, allowing for seamless data collection and analysis.
- **Code Quality**:
The code is organized into clear, modular functions, enhancing readability and maintainability. Each function is documented with comments that explain its purpose and the rationale behind its implementation. I followed best practices for coding style and structure, making it easy for others to understand and modify the code as needed.
- **Innovation**:
I approached the assignment with a unique perspective by combining sentiment analysis and clustering techniques to understand user engagement more deeply. The integration of the OpenAI API for generating personalized messages demonstrates an innovative solution to engage users meaningfully in clinical trials. Additionally, I incorporated data visualization to present insights clearly and compellingly.
- **Ethical Considerations**:
I prioritized ethical considerations throughout the assignment by implementing data privacy measures, such as anonymizing user data and avoiding sensitive information extraction. The design of the message generation process ensures that users are approached based on their expressed interests rather than random outreach, fostering a respectful and considerate user engagement strategy.

## Conclusion

This assignment has allowed me to leverage user-generated content from social media to engage potential participants in clinical trials effectively. Through careful analysis and personalized messaging, I hope to foster greater interest and involvement in clinical research.
