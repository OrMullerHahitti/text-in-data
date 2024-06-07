import praw
import pandas as pd
import numpy as py
import reddit_auth_details

# 1 for Orr and 2 for ibrahim
CurrentUser= reddit_auth_details.RedditUser(2)
CurrentUser.LoadUser(CurrentUser.id)


reddit = praw.Reddit(
    client_id=CurrentUser.client_id,
    client_secret=CurrentUser.client_secret,
    user_agent=CurrentUser.user_agent,
    username=CurrentUser.username,  # Optional
    password=CurrentUser.password,  # Optional
    scopes=['read', 'identity']
)
print(reddit.user.me())


''' all reddit subreddits'''

nootropics = reddit.subreddit('nootropics')
peptides = reddit.subreddit('peptides')
bio_hacks = reddit.subreddit('bioHacks')

# submissions = bio_hacks.search('bpc')
#
# subreddit_name = 'peptides'
# try:
#     subreddit = reddit.subreddit(subreddit_name)
#     subreddit.id  # This will raise an exception if the subreddit doesn't exist
#     print(f"Subreddit '{subreddit_name}' found!")
# except praw.exceptions.Redirect:
#     print(f"Subreddit '{subreddit_name}' not found. Please check the name.")
#     subreddit = None
# except Exception as e:
#     print(f"An error occurred: {e}")
#     subreddit = None
#
# # Define a function to search the subreddit
# def search_subreddit(subreddit, query, limit=100):
#     posts = []
#     if subreddit:
#         try:
#             search_results = subreddit.search(query, limit=limit)
#             for submission in search_results:
#                 posts.append({
#                     'title': submission.title,
#                     'score': submission.score,
#                     'id': submission.id,
#                     'url': submission.url,
#                     'num_comments': submission.num_comments,
#                     'created': submission.created,
#                     'selftext': submission.selftext,
#                     'author': submission.author.name if submission.author else 'N/A'
#                 })
#         except Exception as e:
#             print(f"An error occurred while searching: {e}")
#     return pd.DataFrame(posts)
#
# # Search for 'bpc' in the 'peptides' subreddit if it exists
# if subreddit:
#     search_results = search_subreddit(subreddit, 'bpc', limit=100)
#     if search_results.empty:
#         print("No results found for the query 'bpc'.")
#     else:
#         # Save the search results to a CSV file
#         search_results.to_csv('peptides_subreddit_search_results.csv', index=False)
#         print("Search results collected and saved to 'peptides_subreddit_search_results.csv'.")
# else:
#     print("Search not performed due to subreddit access issue.")