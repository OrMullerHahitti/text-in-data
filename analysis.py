import re
import networkx as nx
import community as community_louvain
from gensim import corpora
from gensim.models import LdaMulticore, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import seaborn as sns
from textblob import TextBlob
from pre_proccessing import preprocess_for_topic_modeling,preprocess_for_sentiment,load_and_preprocess_data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def generate_subject_wordclouds(sentiment_texts, top_subjects):
    for subject, _ in top_subjects:
        # Filter texts for the current subject
        subject_texts = ' '.join([text for subj, text in sentiment_texts if subj == subject])
        words = re.findall(r'\b\w+\b', subject_texts)
        stop_words = set(stopwords.words('english'))
        custom_stop_words = {'would', 'also', 'get', 'im', 'ive', 'take', 'taking', 'day', 'deleted', 'removed', 'yes',
                             'no', 'dont', 'like'}
        stop_words.update(custom_stop_words)
        lemmatizer = WordNetLemmatizer()
        # Remove stopwords
        filtered_words = [word for word in words if word not in stop_words]
        filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]

        # Join the words back into a string
        subject_texts=  ' '.join(filtered_words)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(subject_texts)

        # Create a new figure for each word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for '{subject}'")

        # Save the word cloud
        filename = f"wordcloud_{subject.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()

        print(f"Word cloud for '{subject}' saved as '{filename}'")
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=2)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values
def perform_sna_analysis(graph):
    print("Performing SNA analysis...")

    # Community detection using Louvain method
    partition = community_louvain.best_partition(graph)
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    print("\nDetected communities:")
    for community_id, nodes in communities.items():
        print(f"Community {community_id}: {', '.join(nodes[:5])}{'...' if len(nodes) > 5 else ''}")

    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)

    print("\nTop 5 nodes by degree centrality:")
    print(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

    print("\nTop 5 nodes by betweenness centrality:")
    print(sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

    print("\nTop 5 nodes by closeness centrality:")
    print(sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

    visualize_graph(graph, partition)

def visualize_graph(graph, partition, title="Nootropics and Peptides Interaction Network", node_size_factor=2):
    plt.figure(figsize=(30, 30))
    pos = nx.spring_layout(graph)

    # Calculate node sizes based on degree
    degrees = dict(graph.degree())
    node_sizes = [np.log1p(degrees.get(node, 1))**1.5 * node_size_factor for node in graph.nodes()]

    # Color nodes based on community
    cmap = plt.cm.get_cmap("tab20")
    nx.draw_networkx_nodes(graph, pos, partition.keys(), node_size=node_sizes,
                           cmap=cmap, node_color=list(partition.values()))

    # Draw edges
    edge_weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
    edge_widths = [0.1 + 10 * nw for nw in normalized_weights]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.3, edge_color='gray')

    # Label nodes
    nx.draw_networkx_labels(graph, pos, font_size=2, font_weight='bold')

    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.tight_layout()

    # Add a legend for communities
    unique_communities = set(partition.values())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Community {com}',
                                  markerfacecolor=cmap(com), markersize=10)
                       for com in unique_communities]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('nootropics_peptides_network.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graph visualization saved as 'nootropics_peptides_network.png'")
    print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")


def perform_refined_nlp_analysis(preprocessed_data):
    print("Performing refined NLP analysis...")

    topic_modeling_docs = []
    sentiment_texts = []
    for subject, posts in preprocessed_data.items():
        for post in posts:
            topic_modeling_docs.append(preprocess_for_topic_modeling(post.query) +
                                       preprocess_for_topic_modeling(post.title))
            sentiment_texts.append((subject, preprocess_for_sentiment(post.query)))
            sentiment_texts.append((subject, preprocess_for_sentiment(post.title)))
            for _, comment in post.comments:
                topic_modeling_docs.append(preprocess_for_topic_modeling(comment))
                sentiment_texts.append((subject, preprocess_for_sentiment(comment)))

    # Word frequency analysis using TF-IDF
    dictionary = corpora.Dictionary(topic_modeling_docs)
    corpus = [dictionary.doc2bow(doc) for doc in topic_modeling_docs]
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Calculate average TF-IDF score for each word across all documents
    word_tfidf = {}
    for doc in corpus_tfidf:
        for id, value in doc:
            word = dictionary[id]
            word_tfidf[word] = word_tfidf.get(word, 0) + value

    # Normalize by the number of documents
    for word in word_tfidf:
        word_tfidf[word] /= len(corpus_tfidf)

    # Sort words by their average TF-IDF score
    sorted_word_tfidf = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 most important words (Average TF-IDF):")
    print(sorted_word_tfidf[:10])

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
        dict(sorted_word_tfidf))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud (Average TF-IDF)')
    plt.show()

    # Enhanced Sentiment analysis
    sentiments = [(subject, TextBlob(text).sentiment.polarity) for subject, text in sentiment_texts]
    avg_sentiment = np.mean([s[1] for s in sentiments])
    positive_sentiments = sum(1 for _, s in sentiments if s > 0)
    negative_sentiments = sum(1 for _, s in sentiments if s < 0)
    neutral_sentiments = sum(1 for _, s in sentiments if s == 0)

    print("\nOverall Sentiment Analysis Results:")
    print(f"Average sentiment: {avg_sentiment:.2f}")
    print(f"Positive sentiments: {positive_sentiments} ({positive_sentiments / len(sentiments) * 100:.2f}%)")
    print(f"Negative sentiments: {negative_sentiments} ({negative_sentiments / len(sentiments) * 100:.2f}%)")
    print(f"Neutral sentiments: {neutral_sentiments} ({neutral_sentiments / len(sentiments) * 100:.2f}%)")

    plt.figure(figsize=(10, 5))
    sns.histplot([s[1] for s in sentiments], kde=True)
    plt.title('Overall Sentiment Distribution')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.show()

    # Sentiment analysis by subject
    subject_sentiments = {}
    for subject, sentiment in sentiments:
        if subject not in subject_sentiments:
            subject_sentiments[subject] = []
        subject_sentiments[subject].append(sentiment)

    print("\nSentiment Analysis by Subject:")
    for subject, sentiments_list in subject_sentiments.items():
        avg_sentiment = np.mean(sentiments_list)
        print(f"{subject}: Average sentiment = {avg_sentiment:.2f}")

    # Plot sentiment distribution for top 5 subjects (by number of entries)
    top_subjects = sorted(subject_sentiments.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    generate_subject_wordclouds(sentiment_texts,top_subjects)




    plt.figure(figsize=(12, 8))
    for subject, sentiments_list in top_subjects:
        sns.kdeplot(sentiments_list, label=subject)
    plt.title('Sentiment Distribution for Top 5 Subjects')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Topic modeling
    texts = topic_modeling_docs
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # max_docs = 1000  # adjust as needed
    # texts = texts[:max_docs]
    # corpus = corpus[:max_docs]

    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts, start=5,limit=10, step=1)

    # Find the model with the highest coherence score
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    optimal_topics = optimal_model.num_topics

    print(f"\nOptimal number of topics: {optimal_topics}")
    print("\nDiscovered Topics:")
    for idx, topic in optimal_model.print_topics(-1):
        print(f'Topic {idx}:')
        words = topic.split('+')
        for word in words:
            weight, term = word.split('*')
            print(f'  {term.strip()[1:-1]}: {float(weight):.4f}')
        print()

    # Plot coherence scores
    plt.plot(range(2, 40, 6), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title("Coherence Scores by Number of Topics")
    plt.show()

    return sorted_word_tfidf, sentiments, optimal_model
# The main execution part remains the same
if __name__ == "__main__":
    graph, preprocessed_data = load_and_preprocess_data("data_collected/posts.pkl")
    perform_refined_nlp_analysis(preprocessed_data)
    visualize_graph(graph)
    perform_sna_analysis(graph)

    analyze_graph(graph)