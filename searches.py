import sys
import praw
import reddit_auth_details
import praw
import pandas as pd
import numpy as py
import reddit_auth_details
from main import nootropics, peptides, bio_hacks
import praw
from collections import namedtuple
from collections import Counter



key = namedtuple('key', ['query', 'title', 'comments', 'upvotes', 'author'])


def check_submission(sub1, posts):
    if sub1.id in posts.keys(): return


def search_posts(subreddit, query, subject,posts, sorting="relevance"):
    resaults = subreddit.search(query, limit=100, sort=sorting)
    if subject not in posts:
        posts[subject] = []
    for res in resaults:
        posts[subject].append(())
        print(res.title)


def get_comments(posts, key):
    comments = {}
posts={}
search_posts(peptides, "bpc", "bpc-157",posts)
