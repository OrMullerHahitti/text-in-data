import sys
import praw
import reddit_auth_details
import praw
import pandas as pd
import numpy as py
import reddit_auth_details
from main import  nootropics,peptides,bio_hacks
import praw
posts = {}

def search_posts(subreddit,query,sort="relevance"):
    resaults = subreddit.search(query,limit = 100,sort)
    for res in resaults:
        posts[res.id]=res
        print(res.title)


