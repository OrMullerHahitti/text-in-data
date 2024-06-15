
from search_functions import search_posts, save_posts, load_posts
from main import reddit
from List_of_compounds import peptides, nootropics, genres

''' file to collect the data'''




posts = load_posts("data_collected/posts.pkl")

for peptide in peptides:
    search_posts(reddit, "peptides", peptide, peptide, posts, limit=2)
for nootropic in nootropics:
    search_posts(reddit, "nootropics", nootropic, nootropic, posts, limit=2)
for genre in genres:
    search_posts(reddit, "peptides", genre, genre, posts,limit = 2)
for genre in genres:
    search_posts(reddit, "nootropics", genre, genre, posts,limit = 3)
for genre in genres:
    search_posts(reddit, "bio-hacks", genre, genre, posts,limit = 3)



save_posts(posts, "data_collected/posts.pkl")




