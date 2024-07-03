import networkx as nx
from collections import Counter
#from gensim import corpora
#from gensim.models import LdaModel
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np




def perform_sna_analysis(graph):
    print("Performing SNA analysis...")

    cliques = list(nx.find_cliques(graph))
    print(f"Number of cliques: {len(cliques)}")
    print(f"Largest clique size: {len(max(cliques, key=len))}")

    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)

    print("\nTop 5 nodes by degree centrality:")
    print(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

    print("\nTop 5 nodes by betweenness centrality:")
    print(sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

    print("\nTop 5 nodes by closeness centrality:")
    print(sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

    communities = list(nx.community.greedy_modularity_communities(graph))
    print(f"\nNumber of communities detected: {len(communities)}")

    visualize_graph(graph)


# def perform_nlp_analysis(preprocessed_data):
#     print("Performing NLP analysis...")
#
#     all_texts = []
#     for subject, posts in preprocessed_data.items():
#         for post in posts:
#             all_texts.append(post.query)
#             all_texts.append(post.title)
#             all_texts.extend([comment for _, comment in post.comments])
#
#     word_freq = Counter([word for text in all_texts for word in text])
#
#     print("\nTop 10 most common words:")
#     print(word_freq.most_common(10))
#
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title('Word Cloud')
#     plt.show()
#
#     sentiments = [TextBlob(' '.join(text)).sentiment.polarity for text in all_texts]
#     plt.figure(figsize=(10, 5))
#     plt.hist(sentiments, bins=20)
#     plt.title('Sentiment Distribution')
#     plt.xlabel('Sentiment Polarity')
#     plt.ylabel('Frequency')
#     plt.show()
#
#     dictionary = corpora.Dictionary(all_texts)
#     corpus = [dictionary.doc2bow(text) for text in all_texts]
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42)
#
#     print("\nTop 5 topics:")
#     for idx, topic in lda_model.print_topics(-1):
#         print(f'Topic: {idx} \nWords: {topic}\n')
#



from List_of_compounds import peptidess,racetams,cholinergics,adaptogens,stimulants,amino_acids,herbal_extracts,vitamins_minerals,other_nootropics
def categorize_compound(compound):
    if compound in peptidess:
        return "Peptides"
    elif compound in racetams:
        return "Racetams"
    elif compound in cholinergics:
        return "Cholinergics"
    elif compound in adaptogens:
        return "Adaptogens"
    elif compound in stimulants:
        return "Stimulants"
    elif compound in amino_acids:
        return "Amino Acids"
    elif compound in herbal_extracts:
        return "Herbal Extracts"
    elif compound in vitamins_minerals:
        return "Vitamins and Minerals"
    elif compound in other_nootropics:
        return "Other Nootropics"
    else:
        return "Uncategorized"


def visualize_graph(graph, title="Nootropics and Peptides Interaction Network",node_size_factor=2):
    # Remove isolated nodes
    graph.remove_nodes_from(list(nx.isolates(graph)))

    plt.figure(figsize=(30, 30))

    # Use spring_layout with a small k value to spread out nodes
    pos = nx.spring_layout(graph)

    # Calculate node sizes based on degree
    degrees = dict(graph.degree())
    node_sizes = [np.log1p(degrees.get(node, 1))**1.5 * node_size_factor for node in graph.nodes()]

    # NEW CODE START
    # Categorize nodes and assign colors
    categories = {node: categorize_compound(node) for node in graph.nodes()}
    unique_categories = sorted(set(categories.values()))
    colors = plt.cm.tab20.colors  # This accesses the colors directly
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
    node_colors = [category_colors[categories[node]] for node in graph.nodes()]
    # NEW CODE END


    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

    # Draw edges
    # Extract edge weights
    edge_weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]

    # Normalize edge weights for width
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]

    # Scale edge widths (you can adjust the scaling factor)
    edge_widths = [0.1 + 10 * nw for nw in normalized_weights]

    # Draw edges with varying widths
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    nx.draw_networkx_edges(graph, pos, alpha=0.2, edge_color='gray')

    # Label nodes
    nx.draw_networkx_labels(graph, pos, font_size=2, font_weight='bold')

    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.tight_layout()


    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                                  markerfacecolor=category_colors[cat], markersize=10)
                       for cat in unique_categories]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('nootropics_peptides_network.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graph visualization saved as 'nootropics_peptides_network.png'")
    print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
# Usage remains the same

def analyze_graph(graph):
    # Edge weight analysis
    edge_weights = [d['weight'] for (u, v, d) in graph.edges(data=True)]
    print(f"Edge weight stats:")
    print(f"  Min: {min(edge_weights):.2f}")
    print(f"  Max: {max(edge_weights):.2f}")
    print(f"  Mean: {np.mean(edge_weights):.2f}")
    print(f"  Median: {np.median(edge_weights):.2f}")

    # Node degree analysis
    degrees = [d for n, d in graph.degree()]
    print(f"\nNode degree stats:")
    print(f"  Min: {min(degrees)}")
    print(f"  Max: {max(degrees)}")
    print(f"  Mean: {np.mean(degrees):.2f}")
    print(f"  Median: {np.median(degrees):.2f}")

    # Plotting degree distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20)
    plt.title("Node Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.savefig("degree_distribution.png")
    plt.close()

    # Identify isolated nodes
    isolated_nodes = list(nx.isolates(graph))
    print(f"\nNumber of isolated nodes: {len(isolated_nodes)}")
    if isolated_nodes:
        print(f"Isolated nodes: {isolated_nodes}")

# Usage
if __name__ == "__main__":
    from pre_proccessing import load_and_preprocess_data

    graph, preprocessed_data = load_and_preprocess_data("data_collected/posts.pkl")

    visualize_graph(graph,"checking for the first time")
    perform_sna_analysis(graph)
    analyze_graph(graph)
    #perform_nlp_analysis(preprocessed_data)

