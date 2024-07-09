from flask import Flask, render_template, request, redirect, url_for, session

import pandas as pd
from funciones_recommendation_system import get_datasets, match_researcher_call, recommendation_system_researcher_call, match_call_researcher, recommendation_system_call_researcher
from funciones_filters_recommendation_system import filter_only_publis, filter_only_projects, filter_by_publi_year, filter_by_project_year, filter_by_num_publis, filter_by_num_ip, save_sim_matrix
from funciones_match import agg_mean

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Cambia esto a un valor seguro y aleatorio en producción

# load databases
path = '/Volumes/data_ml4ds/AI4U/Datasets/'
version_wp = '20240510'
version_rp = '20240321'
df_publications, df_projects, df_publications_researchers,df_projects_researchers, df_researchers, df_calls = get_datasets(path, version_wp, version_rp)



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', calls=df_calls.to_dict('records'), researchers=df_researchers.to_dict('records'))

@app.route('/call/<string:call>', methods=['GET', 'POST'])
def call_detail(call):
    call = get_call(call)
    recommending_researchers = False
    recommending_calls = False

    if request.method == 'GET':
        # Limpiar los filtros cuando se carga la página
        session.pop('filters', None)
        return render_template('call_detail.html', call=call, recommendations=None)

    # Procesar la solicitud POST para guardar los filtros
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        page_type = request.form.get('page_type')
        if page_type == 'call_detail':
            recommending_researchers = True
            
        if form_type == 'filters_form':
            filters_dict = {
                'use_publications': 'publications' in request.form.getlist('filters'),
                'use_projects': 'projects' in request.form.getlist('filters'),
                'project_year': request.form['project_year'],
                'publication_year': request.form['publication_year'],
                'num_publications': request.form['num_publications'],
                'num_ip_projects': request.form['num_ip_projects']
            }
            session['filters'] = filters_dict  # Almacenar filtros en la sesión
            return render_template('call_detail.html', call=call, recommendations=None)

        elif form_type == 'methods_form':
            filters_dict = session.get('filters', {})  # Recuperar filtros de la sesión
            recommendation_methods = request.form.getlist('options')
            num_recommendations = int(request.form['number_recommendations'])

            recommendations = get_recommendations(call[0]['Call'], recommendation_methods, num_recommendations, filters_dict, recommending_calls, recommending_researchers)
            return render_template('call_detail.html', call=call, recommendations=recommendations)
    else:
        return render_template('call_detail.html', call=call, recommendations=None)

@app.route('/researcher/<string:researcher_id>', methods=['GET', 'POST'])
def researcher_detail(researcher_id):
    recommending_researchers = False
    recommending_calls = False

    researcher = get_researcher(researcher_id)

    if request.method == 'GET':
        # Limpiar los filtros cuando se carga la página
        session.pop('filters', None)
        return render_template('researcher_detail.html', researcher=researcher, recommendations=None)
    
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        page_type = request.form.get('page_type')

        if page_type == 'researcher_detail':
            recommending_calls = True

        if form_type == 'filters_form':
            filters_dict = {
                'use_publications': 'publications' in request.form.getlist('filters'),
                'use_projects': 'projects' in request.form.getlist('filters'),
                'project_year': '',
                'publication_year': '',
                'num_publications': '',
                'num_ip_projects': ''
            }
            session['filters'] = filters_dict  # Almacenar filtros en la sesión
            return render_template('researcher_detail.html', researcher=researcher, recommendations=None)

        elif form_type == 'methods_form':
            filters_dict = session.get('filters', {})  # Recuperar filtros de la sesión

            recommendation_methods = request.form.getlist('options')
            num_recommendations = int(request.form['number_recommendations'])

            recommendations = get_recommendations(researcher[0]['id_researcher'], recommendation_methods, num_recommendations, filters_dict, recommending_calls, recommending_researchers)
            return render_template('researcher_detail.html', researcher=researcher, recommendations=recommendations)
    else:
        return render_template('researcher_detail.html', researcher=researcher, recommendations=None)
    
@app.route('/filter_calls', methods=['POST'])
def filter_calls():
    '''
    Function for filter the calls
    '''
    cluster_selected = request.form.get('cluster')
    filtered_calls = df_calls[df_calls['Work Programme'] == cluster_selected] if cluster_selected else df_calls


    return render_template('index.html', calls=filtered_calls.to_dict('records'), researchers=df_researchers.to_dict('records'))

@app.route('/filter_researchers', methods=['POST'])
def filter_researchers():
    '''
    Function for filter the researchers
    '''
    department_selected = request.form.get('department')
    filtered_researchers = df_researchers[df_researchers['Department'] == department_selected] if department_selected else df_researchers

    return render_template('index.html', calls=df_calls.to_dict('records'), researchers=filtered_researchers.to_dict('records'))

def get_call(call):
    '''
    Function for obtaining the desired call

    call -> String containing the id of the desired call
    '''
    selected_call = df_calls[df_calls['Call'] == call].to_dict('records')
    return selected_call 

def get_researcher(researcher):
    '''
    Function for obtaining the desired researcher

    researcher -> String containing the id of the desired researcher
    '''
    selected_researcher = df_researchers[df_researchers['id_researcher'] == int(researcher)].to_dict('records')
    return selected_researcher 

def get_recommendations(item, recommendation_method, num_recommendations, filters, recommending_calls, recommending_researchers):
    '''
    Function for obtaining the ranking of the recommended researchers or calls

    item -> id of the desired call or researcher
    recommendation_method -> Selected method for obtaining the recommendations
    num_recommendations -> Number of desired researchers to obtain
    filters -> Dictionary containig the value of the  filters
    '''
    print(filters)

    # Path to sim matrices
    path_sim = '/Volumes/data_ml4ds/AI4U/Datasets/similarity_matrices/publications/'
    save_path = '/Volumes/usuarios_ml4ds/mbalairon/github/recommendation_system_app_dup/sim_matrices/'

    # load researchers database
    df_researchers = pd.read_parquet('/Volumes/data_ml4ds/AI4U/Datasets/ResearchPortal/20240321/parquet/researchers.parquet')
    df_horizon = pd.read_parquet('/Volumes/data_ml4ds/AI4U/Datasets/work_programmes/20240510/horizon_work_programmes.parquet')    
    
    # create sim matrix for researchers and save filtered sim matrix
    df_project_publication_researcher = pd.concat([df_publications_researchers[['id_paper', 'id_researcher']], df_projects_researchers[['actID', 'id_researcher']].rename(columns={'actID':'id_paper'})], ignore_index=True)
    df_project_publication_researcher['id_paper'] = df_project_publication_researcher['id_paper'].apply(convert_to_str)
    df_project_publication_researcher['id_researcher'] = df_project_publication_researcher['id_researcher'].astype(str)

    recommendations = {}

    for method in recommendation_method:
        # load original similarity matrix 
        sim_matrix_publis = pd.read_parquet(path_sim + '{}_sim_matrix.parquet'.format(method))

        # define a name to save the filtered sim matrix
        name = 'similarity_{}_mean.parquet'.format(str(method))
        print('Number of zeros pre filter:', (sim_matrix_publis== 0).sum().sum())
        print('Shape Projects Publis Reserchers pre filter:',df_project_publication_researcher.shape) 

        if len(filters) > 0:  
            # only using publications
            if filters['use_publications'] and not filters['use_projects']:
                sim_matrix_publis = filter_only_publis(sim_matrix_publis)

            # only using projects
            if filters['use_projects'] and not filters['use_publications']:
                sim_matrix_publis = filter_only_projects(sim_matrix_publis)

            # filter by publication year
            if filters['publication_year'] != '':
                pubi_year = int(filters['publication_year'])
                sim_matrix_publis = filter_by_publi_year(pubi_year, sim_matrix_publis, df_publications)

            # filter by project year
            if filters['project_year'] != '':
                project_year = int(filters['project_year'])
                sim_matrix_publis = filter_by_project_year(project_year, sim_matrix_publis, df_projects)

            # filter by number of publis
            if filters['num_publications'] != '':
                num_publis = int(filters['num_publications'])
                df_project_publication_researcher = filter_by_num_publis(num_publis, df_project_publication_researcher, df_researchers)

            # filter by number of projects as principal researcher
            if filters['num_ip_projects'] != '':
                num_ip = int(filters['num_ip_projects'])
                df_project_publication_researcher = filter_by_num_ip(num_ip, df_project_publication_researcher, df_researchers)

        sim_matrix = agg_mean(sim_matrix_publis, df_calls, df_project_publication_researcher) 
        sim_matrix.index = sim_matrix.index.astype(int)      
        save_sim_matrix(sim_matrix, name, save_path)

        if recommending_researchers:
            recommendations['Cosine similarity computed over {}'.format(str(method))] = recommendation_system_call_researcher(method=method, agg_method='mean', call=item, researchers=df_researchers, n=num_recommendations, path=save_path+name).to_dict('records')

        if recommending_calls:
            recommendations['Cosine similarity computed over {}'.format(str(method))] = recommendation_system_researcher_call(method=method, agg_method='mean', researcher=item, calls=df_horizon, n=num_recommendations, path=save_path+name).to_dict('records')
    return recommendations

def convert_to_str(val):
    if isinstance(val, float):
        return str(int(val))  # Elimina el .0 convirtiendo a int primero
    return str(val)  # Deja los strings como están


if __name__ == '__main__':
    app.run(debug=True)