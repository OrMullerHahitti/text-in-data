from collections import namedtuple

key = namedtuple('key', ['query', 'title', 'comments', 'upvotes', 'author'])


def search_posts(subreddit, query, subject, posts, sorting="relevance", limit=10):
    results = subreddit.search(query, limit=limit, sort=sorting)
    if subject not in posts:
        posts[subject] = []
    for res in results:
        if res not in posts[subject]:
            assert isinstance(res, object)
            posts[subject].append(res)
            print(res.title)

