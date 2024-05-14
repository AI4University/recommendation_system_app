import pandas as pd

df_horizon = pd.read_parquet('recommendation_system_app/datasets/horizon_df.parquet')
recommendations = {}

# load the desired similarity measure
similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_embeddings_mean_imp.parquet')
ranking = similarities.loc[researcher].sort_values(ascending=False).fillna(0)
ranking = pd.DataFrame(ranking).reset_index()
id_calls = ranking['index'].to_list()
similarities = ranking[researcher].to_list()
id_calls = pd.DataFrame({'Call': id_calls, 'similarity': similarities})
df_ranking_calls = pd.merge(id_researchers, df_horizon, on='Call', how='inner')
recommendations['Cosine similarity computed over word2vec embeddings'] = df_ranking_calls.head(num_recommendations).to_dict('records')
