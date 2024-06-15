import praw
import pickle
from collections import namedtuple


'''file with the searh functions save , and load the files'''
# Define the Post named tuple
Post = namedtuple('Post', ['query', 'title', 'comments', 'upvotes', 'author'])


# Function to search posts and store them in the posts dictionary
def search_posts(reddit, subreddit_name, query, subject, posts, sorting="relevance", limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    results = subreddit.search(query, limit=limit, sort=sorting)

    if subject not in posts:
        posts[subject] = []

    for res in results:
        if res not in posts[subject]:
            res.comments.replace_more(limit=0)
            comments = [(comment.author.name if comment.author else 'deleted', comment.body) for comment in
                        res.comments.list()]
            post = Post(
                query=res.selftext,  # Assign the selftext attribute to the query field
                title=res.title,
                comments=comments,
                upvotes=res.ups,
                author=res.author.name if res.author else 'deleted'
            )
            posts[subject].append(post)
            print(res.title)


# Function to save the posts dictionary to a file

def save_posts(posts, filename):
    with open(filename, 'wb') as file:
        pickle.dump(posts, file)
    print(f"Posts saved to {filename}")


# Function to load the posts dictionary from a file
def load_posts(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)





