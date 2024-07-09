import pandas as pd
import numpy as np
from tqdm import tqdm


def get_similarity_publication(sim_matrix):
    '''
    Function for getting a dataset with the id of each of the publications and the similarity with each of the calls 
    Pasamos de matriz a dataset
    
    sim_matrix -> similarity matrix 
    '''
    df = sim_matrix.reset_index()

    df['similarity'] = df.apply(lambda row: row.values[1:], axis=1)

    # Seleccionar solo las columnas 'index' e 'similarity'
    df = df[['index', 'similarity']]

    # Renombrar las columnas
    df.columns = ['id', 'similarity']
    
    return df

def agg_mean_imp(sim_matrix, df_calls, df_project_publication_researcher):
    '''
    Function for computing the aggregated of all the similarities of  an author considering only the not thresholded ones
    
    df -> df containing all the similarities without aggreating 
    '''
    # check order of the calls and order them if necessary
    if [str(elemento) for elemento in df_calls['Call'].tolist()] != [str(elemento) for elemento in sim_matrix.columns.tolist()]:
        column_order = [col for col in [str(elemento) for elemento in df_calls['Call'].tolist()] if col in sim_matrix.columns.tolist()]
        sim_matrix = sim_matrix[column_order]

        # elimino las calls que no aparecen en mi sim matrix del dataset con calls
        df_calls['Call'] = df_calls['Call'].astype(str)
        df_calls = df_calls[df_calls['Call'].isin(sim_matrix.columns.tolist())]  

    df = get_similarity_publication(sim_matrix)
    
    df = pd.merge(df_project_publication_researcher, df.rename(columns={'id':'id_paper'}), on='id_paper', how='inner')
    
    result_df = pd.DataFrame(columns=['id_RP', 'similarity'])
    invIDs = df['id_researcher'].unique()

    for invID in tqdm(invIDs, desc="Processing researchers IDs", unit=" id_RP"):
        df_researcher = df[df['id_researcher'] == invID]
        df_vectores_sim = pd.DataFrame(df_researcher['similarity'].tolist())

        suma_columnas = df_vectores_sim.sum()
        non_zero = df_vectores_sim.apply(lambda col: col[col != 0].count())
        mean_imp = (suma_columnas / non_zero).to_list()

        row = pd.DataFrame(data=[[invID, mean_imp]], columns=['id_RP', 'similarity'])
        result_df = pd.concat([result_df, row])

    return get_sim_matrix(df_calls, result_df)

def agg_sum(sim_matrix, df_calls, df_project_publication_researcher):
    '''
    Function for computing the aggregated of all the similarities of  an author considering only the not thresholded ones
    
    df -> df containing all the similarities without aggreating 
    '''
    # check order of the calls and order them if necessary
    if [str(elemento) for elemento in df_calls['Call'].tolist()] != [str(elemento) for elemento in sim_matrix.columns.tolist()]:
        column_order = [col for col in [str(elemento) for elemento in df_calls['Call'].tolist()] if col in sim_matrix.columns.tolist()]
        sim_matrix = sim_matrix[column_order]

        # elimino las calls que no aparecen en mi sim matrix del dataset con calls
        df_calls['Call'] = df_calls['Call'].astype(str)
        df_calls = df_calls[df_calls['Call'].isin(sim_matrix.columns.tolist())]  

    df = get_similarity_publication(sim_matrix)
    df = pd.merge(df_project_publication_researcher, df.rename(columns={'id':'id_paper'}), on='id_paper', how='inner')
    
    result_df = df.groupby('id_researcher')['similarity'].sum().reset_index()
    result_df=result_df.rename(columns= {'id_researcher':'id_RP'})
    
    return get_sim_matrix(df_calls, result_df)

def agg_mean(sim_matrix, df_calls, df_project_publication_researcher):
    '''
    Function for computing the aggregated of all the similarities of  an author considering only the not thresholded ones
    
    df -> df containing all the similarities without aggreating 
    '''

    # check order of the calls and order them if necessary
    if [str(elemento) for elemento in df_calls['Call'].tolist()] != [str(elemento) for elemento in sim_matrix.columns.tolist()]:
        column_order = [col for col in [str(elemento) for elemento in df_calls['Call'].tolist()] if col in sim_matrix.columns.tolist()]
        sim_matrix = sim_matrix[column_order]

        # elimino las calls que no aparecen en mi sim matrix del dataset con calls
        df_calls['Call'] = df_calls['Call'].astype(str)
        df_calls = df_calls[df_calls['Call'].isin(sim_matrix.columns.tolist())]  

    df = get_similarity_publication(sim_matrix)
    df = pd.merge(df_project_publication_researcher, df.rename(columns={'id':'id_paper'}), on='id_paper', how='inner')
        
    result_df = df.groupby('id_researcher')['similarity'].apply(lambda x: pd.Series(x.values.tolist()).mean()).reset_index()
    result_df=result_df.rename(columns= {'id_researcher':'id_RP'})

    return get_sim_matrix(df_calls, result_df)

def get_sim_matrix(df_calls, df_researchers):
    '''
    Function for computing the similarity matrix for the researchers and the calls
    Pasamos de dataset a matriz 
    
    df_calls -> dataframe with all the avaliable calls
    df_researchers -> Dataframe containing the researchers and the similarity with each of the publications
    '''

    keys_calls = df_calls['Call'].tolist()
    keys_res = df_researchers['id_RP'].tolist()
    sim = df_researchers['similarity'].tolist()
    
    df = pd.DataFrame(sim, columns=keys_calls, index=keys_res)
    
    return df