def generate_word_clouds(preprocessed_data, num_subjects=5):
    # Count the number of posts for each subject
    subject_sizes = {subject: len(posts) for subject, posts in preprocessed_data.items()}

    # Get the top subjects
    top_subjects = sorted(subject_sizes.items(), key=lambda x: x[1], reverse=True)[:num_subjects]

    for subject, size in top_subjects:
        # Combine all text for the subject
        all_text = []
        for post in preprocessed_data[subject]:
            all_text.extend(post.query)
            all_text.extend(post.title)
            for _, comment in post.comments:
                all_text.extend(comment)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_text))

        # Create a new figure for each word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for '{subject}' (Posts: {size})")

        # Save the individual word cloud
        filename = f"wordcloud_{subject.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

        # Show the plot (this will open a new window for each)
        plt.show()

        print(f"Word cloud for '{subject}' saved as '{filename}'")

    print(f"Generated word clouds for top {num_subjects} subjects")