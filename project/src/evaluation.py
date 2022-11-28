import sklearn
from project import *
from sklearn.mixture import GaussianMixture
from project.utils.plotting import plot_bar_graph

def cluster_distributional_analysis(embeddings_list,no_of_clusters,model_name,column):
    cluster_dict = multi_cluster_analysis(embeddings_list,no_of_clusters)
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
    plot_bar_graph(langauge,cluster_score,model_name,column)


def multi_cluster_analysis(embeddings_list,no_of_clusters):
    gmm = GaussianMixture(n_components=no_of_clusters,random_state=42,init_params="k-means++").fit(embeddings_list[0])
    print(gmm.means_)

    lang_dict = {0:"english",1:"french", 2:"spanish",3:"dutch",4:"german"}
    final_dict = {}
    for i in range(0,len(embeddings_list)):
        predict = gmm.predict(embeddings_list[i])
        for j in range(len(embeddings_list[0])):
            if( j in final_dict):
                if(len(final_dict)>0):
                    #print("in else")
                    #final_dict[j] = {}
                    final_dict[j][lang_dict[i]]= predict[j]
                else:
                    final_dict[j][lang_dict[i]] = predict[j]
                #print(j,lang_dict[i],predict[j])
            else:
                final_dict[j] = {}
                final_dict[j][lang_dict[i]] = predict[j]
            #print(final_dict[j])
    #print(final_dict)
    return final_dict




