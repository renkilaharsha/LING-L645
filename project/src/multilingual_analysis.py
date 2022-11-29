import pandas as pd
import matplotlib.pyplot as plt
from numpy import dot
from ast import literal_eval
import numpy as np
import seaborn as sns
from project import *



class MultilingualAnalysis:
    def __init__(self,path):
        self.path = path

    def cosine_similarity(self,a, b):
        return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))

    def vectorspace_analysis(self,language,column):
        langs = ["dutch","spanish","german","french"]

        df_lang = pd.read_csv(self.path.format(language))
        #print(df_lang.columns)
        cossim_dict = {}
        for lang in langs:
            if(lang!= language):
                cossim_dict["{}_{}_{}".format(language, lang,column)] = []
                df = pd.read_csv(self.path.format(lang))
                #print(df.columns)
                #cossim_dict["{}_{}_{}".format(language, lang, column)] = cosine_similarity(literal_eval(df["{}_{}_embeddings".format(lang,column)]),literal_eval(df_lang["{}_{}_embeddings".format(language,column)]))
                for i in range(len(df)):
                   cossim_dict["{}_{}_{}".format(language, lang,column)].append(self.cosine_similarity(np.array(literal_eval(df["{}_{}_embeddings".format(lang,column)].iloc[i])),np.array(literal_eval(df_lang["{}_{}_embeddings".format(language,column)].iloc[i]))))
        self.plot_density_distribution(cossim_dict,"/Users/harsharenkila/PycharmProjects/ANLP/project/output/cosine_sim_analysis/{}_{}.png".format("M_USE",column),camel_case(column))
        return cossim_dict

    def plot_density_distribution(self,cossim_dict,path,column):
        keys = cossim_dict.keys()
        for key in keys:
            # Subset to the airline


            # Draw the density plot
            sns.distplot(cossim_dict[key], hist=False, kde=True,
                         kde_kws={'linewidth': 3},
                         label=key)

        # Plot formatting
        plt.legend(prop={'size': 16}, title='Language pair')
        plt.title('{} Density Plot with Language-pair cosine similarity'.format(column))
        plt.xlabel('cosine similarity')
        plt.ylabel('Density')
        plt.savefig(path)
        plt.show()

path = "/Users/harsharenkila/PycharmProjects/ANLP/project/embeddings/M_USE_{}_data.csv"
kk = MultilingualAnalysis(path)

kk.vectorspace_analysis("english","title")
kk.vectorspace_analysis("english","description")
kk.vectorspace_analysis("english","domain")