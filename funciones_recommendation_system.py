import pandas as pd
import numpy as np

def get_datasets(path, version_wp, version_rp):
    '''
    Function for importing all the necessary databases for the recommendation system
    
    path -> General path to the Databases storage
    
    df_publications -> Dataset containg all the data related to the publications
    df_publications_researchers -> Dataset containg all the relationships of researchers and publications
    df_researchers -> Dataset containg all the data related to the researchers 
    df_calls -> Dataset containg all the data related to the calls
    '''
    
    df_eic = pd.read_parquet(path + 'work_programmes/20231001/EIC_work_programmes.parquet')
    df_horizon = pd.read_parquet(path +'work_programmes/{}/horizon_work_programmes.parquet'.format(version_wp))

    df_publications = pd.read_parquet(path + 'ResearchPortal/{}/parquet/publications.parquet'.format(version_rp))
    df_publications_researchers = pd.read_parquet(path + 'ResearchPortal/{}/parquet/researchers_publications.parquet'.format(version_rp))
    df_projects = pd.read_parquet(path + 'ResearchPortal/{}/parquet/projects.parquet'.format(version_rp))
    df_projects_researchers = pd.read_parquet(path + 'ResearchPortal/{}/parquet/researchers_projects.parquet'.format(version_rp))
    df_researchers = pd.read_parquet(path + 'ResearchPortal/{}/parquet/researchers.parquet'.format(version_rp))

    
    # join together al the calls (igual que cuando creamos las matrices de similitud)
    df_eic['Call'] = df_eic['id']
    df_eic['Title'] = df_eic['Challenge']

    df_calls = pd.concat([df_horizon[['Call', 'Work Programme', 'Title']], df_eic[['Call', 'Work Programme', 'Title']]], axis=0).reset_index(drop=True)

    
    return df_publications, df_projects, df_publications_researchers,df_projects_researchers, df_researchers, df_calls

def match_researcher_call(similarities, researcher, df_calls, n=766):
    '''
    Function for obtaining the ranking of calls given a researcher 
    
    similarities -> df with all the smmilarities between researchers and calls
    researcher -> Researcher of interest
    n -> Number of recommendations we are interested in 
    df_calls -> Dataframe with the information about the calls
    
    df_ranking_calls -> Dataset with the ranking of recommended calls for a given researcher
    '''
    
    ranking = similarities.transpose()[researcher].sort_values(ascending=False).fillna(0)
    ranking = pd.DataFrame(ranking).reset_index()
    id_calls = ranking['index'].to_list()
    similarities = ranking[researcher].to_list()
    id_calls = pd.DataFrame({'Call': id_calls, 'similarity': similarities})
    df_ranking_calls = pd.merge(id_calls, df_calls, on='Call', how='inner')

    return df_ranking_calls.head(n)

def recommendation_system_researcher_call(method, agg_method, researcher, calls, n=766,
                         path='/export/data_ml4ds/AI4U/Datasets/similarity_matrices/researchers/similarity_{}_{}.parquet'):
    '''
    function for obtaining the recommendations of calls given a researcher
    
    path -> Path to the file containing the similarity matrix. Important to have the similarity matrices stored as similarity_method_aggMethod.parquet
    method -> Method selected to calculate the similarities 
    agg_method -> Agregation method selected for calculating the similarties between calls and researchers
    '''
    
    similarities = pd.read_parquet(path.format(method, agg_method))
    ranking = match_researcher_call(similarities, researcher, calls, n)
    #return ranking[['Call', 'Work Programme', 'similarity']]
    return ranking

def match_call_researcher(similarities, call, df_researchers, n=1227):
    '''
    Function for obtaining the ranking of researchers given a call 
    
    similarities -> df with all the smmilarities between researchers and calls
    call -> Call of interest
    n -> Number of researchers we are interested in 
    df_researchers -> Dataframe with the information about the researchers
    
    df_ranking_researchers -> Dataset with the ranking of recommended researchers for a given call
    '''
    
    ranking = similarities[call].sort_values(ascending=False).fillna(0)
    ranking = pd.DataFrame(ranking).reset_index()
    id_researchers = ranking['index'].to_list()
    similarities = ranking[call].to_list()
    id_researchers = pd.DataFrame({'id_researcher': id_researchers, 'similarity': similarities})
    df_ranking_researchers = pd.merge(id_researchers, df_researchers, on='id_researcher', how='inner')

    return df_ranking_researchers.head(n)

def recommendation_system_call_researcher(method, agg_method, call, researchers, n=1227,
                         path='/export/data_ml4ds/AI4U/Datasets/similarity_matrices/researchers/similarity_{}_{}.parquet'):
    '''
    function for obtaining the recommendations of researchers given a call
    
    path -> Path to the file containing the similarity matrix. Important to have the similarity matrices stored as similarity_method_aggMethod.parquet
    method -> Method selected to calculate the similarities 
    agg_method -> Agregation method selected for calculating the similarties between calls and researchers
    '''
    
    similarities = pd.read_parquet(path.format(method, agg_method))
    ranking = match_call_researcher(similarities, call, researchers, n)
    #return ranking[['invID', 'Department', 'Research Group', 'Subjects', 'no Publis', 'similarity']]
    return ranking

