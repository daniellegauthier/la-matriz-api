import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

def analyze_sequences(colors_df, sequences_df, semantic_mapping_df, k=5):
    output = {}

    # Prepare and clean data
    for col in ['r', 'g', 'b']:
        colors_df[col] = pd.to_numeric(colors_df[col], errors='coerce')
    for col in ['R', 'G', 'B']:
        semantic_mapping_df[col] = pd.to_numeric(semantic_mapping_df[col], errors='coerce')

    # Vectorize text
    colors_df['clean_words'] = colors_df['english-words'].fillna('').astype(str)
    vectorizer = CountVectorizer(binary=True, max_features=100)
    sentiment_features = vectorizer.fit_transform(colors_df['clean_words'])
    sentiment_df = pd.DataFrame(
        sentiment_features.toarray(),
        columns=[f'sentiment_{word}' for word in vectorizer.get_feature_names_out()],
        index=colors_df.index
    )

    # Combine features
    X_rgb = colors_df[['r', 'g', 'b']].fillna(0)
    X = pd.concat([X_rgb, sentiment_df], axis=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    colors_df['cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for visualization (optional)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Build cluster summaries
    cluster_summary = {}
    for cluster in range(k):
        cluster_colors = colors_df[colors_df['cluster'] == cluster]
        top_colors = cluster_colors['color'].value_counts().head(5).index.tolist()
        avg_rgb = cluster_colors[['r', 'g', 'b']].mean()
        cluster_summary[cluster] = {
            "count": len(cluster_colors),
            "top_colors": top_colors,
            "avg_rgb": avg_rgb.round(1).to_dict()
        }

    output['clusters'] = cluster_summary

    # Map colors to clusters
    color_to_cluster = dict(zip(colors_df['color'], colors_df['cluster']))

    # Parse sequences
    def parse_sequence(seq_str):
        if pd.isna(seq_str):
            return []
        return [c.strip().lower() for c in seq_str.split(',')]

    sequences_df['parsed_sequence'] = sequences_df['sequence'].apply(parse_sequence)

    # Calculate sequence momentum
    def calculate_momentum(sequence, color_to_cluster):
        momentum = 0
        for i in range(len(sequence) - 1):
            c1 = color_to_cluster.get(sequence[i])
            c2 = color_to_cluster.get(sequence[i+1])
            if c1 is not None and c2 is not None:
                momentum += int(c1 != c2)
        return momentum / max(1, len(sequence) - 1)

    sequences_df['momentum'] = sequences_df['parsed_sequence'].apply(
        lambda seq: calculate_momentum(seq, color_to_cluster)
    )

    # Top sequences with highest momentum
    top_sequences = sequences_df.sort_values('momentum', ascending=False).head(5)

    output['top_sequences'] = top_sequences[['sequence', 'momentum']].to_dict(orient='records')

    return output
