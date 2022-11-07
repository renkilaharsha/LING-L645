
import pandas as pd
from project import *

def preprocess_data(occupation_path, job_zone_path):
    occupation_df = pd.read_excel(occupation_path)
    job_zone_df = pd.read_excel(job_zone_path)
    jobzone_df = job_zone_df.fillna(0)
    df_merge = pd.merge(jobzone_df, occupation_df, on=['O*NET-SOC Code','Title'], how='inner')

    domain_list= []
    for i in range(len(df_merge)):
        k = int(df_merge["O*NET-SOC Code"].iloc[i].split("-")[0])
        domain_list.append(domain[k])

    df_merge["Domain"] = domain_list
    df_merge.to_csv("project/data/Processed_data.csv")