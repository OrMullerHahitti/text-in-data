from main import peptides
from searches import search_posts

posts = {}
search_posts(peptides, "tb-500", "tb-500", posts,limit=100)
search_posts(peptides, "bpc", "bpc-157", posts,limit=100)
search_posts(peptides, "best peptide for cognition", "cognition", posts, "hot")
search_posts(peptides, "best peptide for longevity", "longevity", posts, "hot", 100)

search_posts(peptides, "best peptide for injury", "healing", posts, "hot", 100)

search_posts(peptides, "best peptide for healing", "mental health", posts, "hot",100)

search_posts(peptides, "best peptide for libido", "libido", posts, "hot",100)

search_posts(peptides, "best peptide for fat loss", "fat loss", posts, "hot",100)
search_posts(peptides, "best peptide for health", "general health", posts, "hot",100)

search_posts(peptides, "best peptide for healing", "healing", posts, "hot",100)
