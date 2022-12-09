from project import *
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from project.utils.plotting import plot_bar_graph

def cluster_distributional_analysis(embeddings_list,no_of_clusters,model_name,column,clustering_model=None):
    cluster_dict = multi_cluster_analysis(embeddings_list,no_of_clusters,clustering_model)
    language_pairs = [(a, b) for idx, a in enumerate(langauges) for b in langauges[idx + 1:]]
    langauge = []
    cluster_score=[]
    for lang_a,lang_b in language_pairs:
        if(language_code[lang_a]+"_"+language_code[lang_b] or language_code[lang_b]+"_"+language_code[lang_a]) not in langauge:
            langauge.append(language_code[lang_a]+"_"+language_code[lang_b])
            count = 0
            for i in cluster_dict:
                if(cluster_dict[i][lang_a]==cluster_dict[i][lang_b]):
                    count+=1
            cluster_score.append(count)
    plot_bar_graph(langauge,cluster_score,model_name,column,clustering_model)


def multi_cluster_analysis(embeddings_list,no_of_clusters,model:str=None):
    if(model == None):
        clustering = GaussianMixture(n_components=no_of_clusters,random_state=0,init_params="k-means++").fit(embeddings_list[0])
    else:
        clustering = KMeans(n_clusters=no_of_clusters,init="k-means++",random_state=0).fit(embeddings_list[0])
        #clustering =SpectralClustering(n_clusters=no_of_clusters,assign_labels="discretize",random_state=0).fit(embeddings_list[0])
    lang_dict = {0:"english",1:"french", 2:"spanish",3:"dutch",4:"german"}
    final_dict = {}
    print(model)
    for i in range(0,len(embeddings_list)):
        predict = clustering.predict(embeddings_list[i])
        for j in range(len(embeddings_list[0])):
            if( j in final_dict):
                if(len(final_dict)>0):
                    #print("in else")
                    #final_dict[j] = {}
                    final_dict[j][lang_dict[i]] = predict[j]
                else:
                    final_dict[j][lang_dict[i]] = predict[j]
                #print(j,lang_dict[i],predict[j])
            else:
                final_dict[j] = {}
                final_dict[j][lang_dict[i]] = predict[j]
            #print(final_dict[j])
    #print(final_dict)
    return final_dict




