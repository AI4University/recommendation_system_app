import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from funciones_match import get_similarity_publication, get_sim_matrix, agg_mean_imp, agg_sum, agg_mean
from funciones_recommendation_system import get_datasets, match_researcher_call, recommendation_system_researcher_call, match_call_researcher, recommendation_system_call_researcher

# define filter functions:
def filter_only_publis(sim_matrix):
    '''
    Function for filtering out all the projects and use only the publications fot generating recommendations

    sim_matrix -> Similarity matrix to filter
    '''
    sim_matrix[sim_matrix.index.str.startswith('act')] = 0
    return sim_matrix

def filter_only_projects(sim_matrix):
    '''
    Function for filtering out all the publications and use only the projects fot generating recommendations

    sim_matrix -> Similarity matrix to filter
    '''
    rows_to_zero = sim_matrix.index[~sim_matrix.index.str.startswith('act')]
    sim_matrix.loc[rows_to_zero] = 0
    return sim_matrix

def filter_by_publi_year(year, sim_matrix, df_publications):
    '''
    Function for filtering a similarity matrix based on the publication year of publication

    year -> Year from which we apply the filter
    sim_matrix -> Similarity matrix to filter
    df_publications -> Dataset containing all the publications
    '''
    filtered_publications = df_publications[df_publications['year'] < float(year)]
    delete_ids = list(filtered_publications['id_paper'])
    existing_ids = [id for id in delete_ids if id in sim_matrix.index]
    sim_matrix.loc[existing_ids] = 0
    return sim_matrix

def filter_by_project_year(year, sim_matrix, df_projects):
    '''
    Function for filtering a similarity matrix based on the ending year of the project

    year -> Year from which we apply the filter
    sim_matrix -> Similarity matrix to filter
    df_projects -> Dataset containing all the projects
    '''

    filtered_projects = df_projects[df_projects['EndYear'] < year]
    delete_ids = list(filtered_projects['actID'])
    existing_ids = [id for id in delete_ids if id in sim_matrix.index]
    sim_matrix.loc[existing_ids] = 0
    return sim_matrix

def filter_by_num_publis(minimum_publications, df_project_publication_researcher, df_researchers):
    '''
    Function for filtering out thous researchers without a minimum number of publications
    minimum_publications -> Minimum number of publications that a researcher must have
    df_project_publication_researcher -> Dataset containg all the relations between resarchers and publications and projects
    df_researchers -> Dataset containing all the researchers data
    '''
    
    filtered_researchers = df_researchers[df_researchers['no_publis'] < minimum_publications]
    delete_ids = list(filtered_researchers['id_researcher'])
    delete_ids = list(map(str, delete_ids))
    df_project_publication_researcher = df_project_publication_researcher[~df_project_publication_researcher['id_researcher'].isin(delete_ids)]

    return df_project_publication_researcher

def filter_by_num_ip(minimum_ips, df_project_publication_researcher, df_researchers):
    '''
    Function for filtering out thous researchers without a minimum number of projects as principal researchers
    minimum_ips -> Minimum number of projects as principal researcher that a researcher must have
    df_project_publication_researcher -> Dataset containg all the relations between resarchers and publications and projects
    df_researchers -> Dataset containing all the researchers data
    '''
    
    filtered_researchers = df_researchers[df_researchers['Projects_IP'] < minimum_ips]
    delete_ids = list(filtered_researchers['id_researcher'])
    delete_ids = list(map(str, delete_ids))
    df_project_publication_researcher = df_project_publication_researcher[~df_project_publication_researcher['id_researcher'].isin(delete_ids)]

    return df_project_publication_researcher

def save_sim_matrix(sim_matrix, name, path):
    '''
    function for saving the similarity matrix filtered
    
    sim_matrix -> Similarity matrix filtered
    name -> Name to save the similarity matrix
    path -> Saving path
    '''

    sim_matrix.to_parquet(path+name)




