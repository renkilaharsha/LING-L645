import pandas as pd
import ast
import numpy as np


def get_visualize_data(path:str="/Users/harsharenkila/PycharmProjects/ANLP/project/embeddings/",model:str="Xlmr_Bert", language:str="english",column:str="title"):
    df = pd.read_csv(path+"{}_{}_data.csv".format(model,language),index_col=None)
    embedding_list = extract_embeddings(df["{}_{}_embeddings".format(language,column)])
    if(column == "title" and language=="english"):
        return df["Title"].tolist(),embedding_list
    if (column == "description" and language == "english"):
        return df["Description"].tolist(), embedding_list
    if (column == "domain" and language == "english"):
        return df["Domain"].tolist(), embedding_list
    return df["{}_{}".format(language,column)].tolist(),embedding_list

def get_data_from_file(path:str = "/Users/harsharenkila/PycharmProjects/ANLP/project/embeddings/",model:str="Xlmr_Bert", language:str="english"):
    df = pd.read_csv(path+"{}_{}_data.csv".format(model,language),index_col=None)
    title_embedding_list = extract_embeddings(df["{}_title_embeddings".format(language)])
    description_embedding_list = extract_embeddings(df["{}_title_embeddings".format(language)])
    domain_embedding_list = extract_embeddings(df["{}_title_embeddings".format(language)])
    job_zone = df["Job Zone"].tolist()
    if(len(title_embedding_list)== len(df) and len(description_embedding_list)== len(df) and len(domain_embedding_list)== len(df) and len(job_zone)== len(df)):
        title_embedding = np.array(title_embedding_list,np.float32)
        description_embedding = np.array(description_embedding_list,np.float32)
        domain_embedding =  np.array(domain_embedding_list,np.float32)
        return title_embedding,description_embedding,domain_embedding,job_zone

    return False

def upsample_data(job_zone,title,description):
    pass

def extract_embeddings(df_column):
    data = []
    for i in range(len(df_column)):
        try:
            data.append(ast.literal_eval(df_column.iloc[i]))
        except:
            print("Error while getting the data from csv : {} ".format(i))
    return np.array(data)