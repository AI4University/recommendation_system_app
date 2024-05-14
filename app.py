from flask import Flask, render_template, request, redirect, url_for

import pandas as pd


app = Flask(__name__)

# load database
df_horizon = pd.read_parquet('recommendation_system_app/datasets/horizon_df.parquet')
df_researchers = pd.read_parquet('recommendation_system_app/datasets/researchers.parquet')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'cluster' in request.form:
            cluster_selected = request.form['cluster']
            call_selected = None

        elif 'callId' in request.form:
            cluster_selected = None
            call_selected = request.form['callId']

        if cluster_selected: 
            horizon_filtered = filter_calls(cluster_selected).to_dict('records')

        if call_selected: 
            horizon_filtered = df_horizon[df_horizon['Call']==call_selected].to_dict('records')

        return render_template('index.html', calls=horizon_filtered)
    else:
        return render_template('index.html', calls = df_horizon.to_dict('records'))

@app.route('/call/<string:call>', methods=['GET', 'POST'])
def call_detail(call):
    call = get_call(call)

    if request.method == 'POST':
        recommendation_methods = request.form.getlist('options')
        num_recommendations = int(request.form['number_recommendations'])
        recommendations = get_recommendations(call[0]['Call'], recommendation_methods, num_recommendations)
        return render_template('call_detail.html', call=call, recommendations=recommendations)
    else:
        return render_template('call_detail.html', call=call, recommendations=None)

@app.route('/searchByResearcher', methods=['GET', 'POST'])
def search_by_researcher():
    if request.method == 'POST':
        department_selected = request.form.get('department')
        if department_selected:
            df_researchers = filter_researchers(department_selected)
        else: 
            df_researchers = pd.read_parquet('recommendation_system_app/datasets/researchers.parquet')

        return render_template('searchByResearcher.html', researchers=df_researchers.to_dict('records'))

    else:
        return render_template('searchByResearcher.html', researchers=df_researchers.to_dict('records'))

@app.route('/researcher/<string:researcher>', methods=['GET', 'POST'])
def researcher_detail(researcher):
    researcher = get_researcher(researcher)

    if request.method == 'POST':
        recommendation_methods = request.form.getlist('options')
        num_recommendations = int(request.form['number_recommendations'])
        recommendations = get_recommendations_calls(researcher[0]['invID'], recommendation_methods, num_recommendations)
        return render_template('researcher_detail.html', researcher=researcher, recommendations=recommendations)
    else:
        return render_template('researcher_detail.html', researcher=researcher, recommendations=None)
    
def filter_calls(cluster):
    '''
    Function for filter the calls given a cluster

    cluster -> String containing the desired cluster
    '''
    return df_horizon[df_horizon['Work Programme'] == cluster]

def filter_researchers(department):
    '''
    Function for filter the researchers given a department

    department -> String containing the desired department
    '''
    return df_researchers[df_researchers['Department'] == department]

def get_call(call):
    '''
    Function for obtaining the desired call

    call -> String containing the id of the desired call
    '''
    selected_call = df_horizon[df_horizon['Call'] == call].to_dict('records')
    return selected_call 

def get_researcher(researcher):
    '''
    Function for obtaining the desired researcher

    researcher -> String containing the id of the desired researcher
    '''
    selected_researcher = df_researchers[df_researchers['invID'] == researcher].to_dict('records')
    return selected_researcher 

def get_recommendations(call, recommendation_method, num_recommendations):
    '''
    Function for obtaining the ranking of the recommended researchers

    call -> id of the desired call
    recommendation_method -> Selected method for obtaining the recommendations
    num_recommendations -> Number of desired researchers to obtain
    '''

    # load researchers database
    df_researchers = pd.read_parquet('data_ingest/UC3M ResearchPortal/Outputs/researchers.parquet')
    recommendations = {}
    # load the desired similarity measure
    if 'word2vec' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_embeddings_mean_imp.parquet')
        ranking = similarities[call].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_researchers = ranking['index'].to_list()
        similarities = ranking[call].to_list()
        id_researchers = pd.DataFrame({'invID': id_researchers, 'similarity': similarities})
        df_ranking_researchers = pd.merge(id_researchers, df_researchers, on='invID', how='inner')
        recommendations['Cosine similarity computed over word2vec embeddings'] = df_ranking_researchers.head(num_recommendations).to_dict('records')

    if 'bert' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_BERT_mean_imp.parquet')
        ranking = similarities[call].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_researchers = ranking['index'].to_list()
        similarities = ranking[call].to_list()
        id_researchers = pd.DataFrame({'invID': id_researchers, 'similarity': similarities})
        df_ranking_researchers = pd.merge(id_researchers, df_researchers, on='invID', how='inner')
        recommendations['Cosine similarity computed over BERT embeddings'] = df_ranking_researchers.head(num_recommendations).to_dict('records')

    if 'tf-idf' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_tfidf_mean_imp.parquet')
        ranking = similarities[call].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_researchers = ranking['index'].to_list()
        similarities = ranking[call].to_list()
        id_researchers = pd.DataFrame({'invID': id_researchers, 'similarity': similarities})
        df_ranking_researchers = pd.merge(id_researchers, df_researchers, on='invID', how='inner')
        recommendations['Cosine similarity computed over TF-IDF vectorizations'] = df_ranking_researchers.head(num_recommendations).to_dict('records')

    if 'bhattacharyya' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_bhattacharyya_mean_imp.parquet')
        ranking = similarities[call].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_researchers = ranking['index'].to_list()
        similarities = ranking[call].to_list()
        id_researchers = pd.DataFrame({'invID': id_researchers, 'similarity': similarities})
        df_ranking_researchers = pd.merge(id_researchers, df_researchers, on='invID', how='inner')
        recommendations['Bhattacharyya similarity of topics distributions'] = df_ranking_researchers.head(num_recommendations).to_dict('records')


    return recommendations

def get_recommendations_calls(researcher, recommendation_method, num_recommendations):
    '''
    Function for obtaining the ranking of the recommended calls

    researcher -> id of the desired researcher
    recommendation_method -> Selected method for obtaining the recommendations
    num_recommendations -> Number of desired researchers to obtain
    '''

    # load researchers database
    df_horizon = pd.read_parquet('recommendation_system_app/datasets/horizon_df.parquet')
    recommendations = {}

    # load the desired similarity measure
    if 'word2vec' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_embeddings_mean_imp.parquet')
        ranking = similarities.loc[researcher].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_calls = ranking['index'].to_list()
        similarities = ranking[researcher].to_list()
        id_calls = pd.DataFrame({'Call': id_calls, 'similarity': similarities})
        df_ranking_calls = pd.merge(id_calls, df_horizon, on='Call', how='inner')
        recommendations['Cosine similarity computed over word2vec embeddings'] = df_ranking_calls.head(num_recommendations).to_dict('records')

    if 'bert' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_BERT_mean_imp.parquet')
        ranking = similarities.loc[researcher].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_calls = ranking['index'].to_list()
        similarities = ranking[researcher].to_list()
        id_calls = pd.DataFrame({'Call': id_calls, 'similarity': similarities})
        df_ranking_calls = pd.merge(id_calls, df_horizon, on='Call', how='inner')
        recommendations['Cosine similarity computed over BERT embeddings'] = df_ranking_calls.head(num_recommendations).to_dict('records')

    if 'tf-idf' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_tfidf_mean_imp.parquet')
        ranking = similarities.loc[researcher].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_calls = ranking['index'].to_list()
        similarities = ranking[researcher].to_list()
        id_calls = pd.DataFrame({'Call': id_calls, 'similarity': similarities})
        df_ranking_calls = pd.merge(id_calls, df_horizon, on='Call', how='inner')
        recommendations['Cosine similarity computed over TF-IDF vectorizations'] = df_ranking_calls.head(num_recommendations).to_dict('records')

    if 'bhattacharyya' in recommendation_method:
        similarities = pd.read_parquet('recommendation_system_app/simmilarity_matrices/similarity_bhattacharyya_mean_imp.parquet')
        ranking = similarities.loc[researcher].sort_values(ascending=False).fillna(0)
        ranking = pd.DataFrame(ranking).reset_index()
        id_calls = ranking['index'].to_list()
        similarities = ranking[researcher].to_list()
        id_calls = pd.DataFrame({'Call': id_calls, 'similarity': similarities})
        df_ranking_calls = pd.merge(id_calls, df_horizon, on='Call', how='inner')
        recommendations['Bhattacharyya similarity of topics distributions'] = df_ranking_calls.head(num_recommendations).to_dict('records')

    return recommendations
if __name__ == '__main__':
    app.run(debug=True)