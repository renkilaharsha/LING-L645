import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio


from sklearn.manifold import TSNE
from project.utils.data_operations import *

def plot_training_val_curves(history,model_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("/Users/harsharenkila/PycharmProjects/ANLP/project/output/loss_plots/{}_loss_plot".format(model_name))
    plt.show()


def get_sample_index():
    df = pd.read_csv("/Users/harsharenkila/PycharmProjects/ANLP/project/embeddings/FastText_dutch_data.csv")
    index = df.sample(5, replace=False).index
    del df
    return index
#https://medium.com/analytics-vidhya/word-embedding-using-python-63770334841
def visualize_word_vectors(model_name,column,index):

    eng_txt,eng_embedding = get_visualize_data(model=model_name,language="english",column=column)
    spa_txt,spa_embedding = get_visualize_data(model=model_name,language="spanish",column=column)
    ger_txt,ger_embedding = get_visualize_data(model=model_name,language="german",column=column)
    fre_txt,fre_embedding = get_visualize_data(model=model_name,language="french",column=column)
    dut_txt,dut_embedding = get_visualize_data(model=model_name,language="dutch",column=column)

    pca = TSNE(n_components=2, learning_rate='auto',init = 'random', perplexity = 3)
    #pca = PCA(n_components=2)

    eng = pca.fit_transform(eng_embedding)
    spa = pca.fit_transform(spa_embedding)
    ger = pca.fit_transform(ger_embedding)
    fre = pca.fit_transform(fre_embedding)
    dut = pca.fit_transform(dut_embedding)


    eng_df = pd.DataFrame(eng, columns=list('XY'))
    # adding a columns for the corresponding words
    eng_df['Words'] = eng_txt
    eng_df['Language'] = ["English"]*len(eng_df)

    spa_df = pd.DataFrame(spa, columns=list('XY'))
    # adding a columns for the corresponding words
    spa_df['Words'] = spa_txt
    spa_df['Language'] = ["Spanish"]*len(eng_df)


    ger_df = pd.DataFrame(ger, columns=list('XY'))
    # adding a columns for the corresponding words
    ger_df['Words'] = ger_txt
    ger_df['Language'] = ["German"]*len(eng_df)


    fre_df = pd.DataFrame(fre, columns=list('XY'))
    # adding a columns for the corresponding words
    fre_df['Words'] = fre_txt
    fre_df['Language'] = ["French"]*len(eng_df)



    dut_df = pd.DataFrame(dut, columns=list('XY'))
    # adding a columns for the corresponding words
    dut_df['Words'] = dut_txt
    dut_df['Language'] = ["Dutch"]*len(eng_df)

    eng_df = eng_df.loc[index]
    df =  pd.concat([eng_df.loc[index],spa_df.loc[index],ger_df.loc[index],fre_df.loc[index],dut_df.loc[index]],axis=0, ignore_index=True)

    # plotting a scatter plot
    fig = px.scatter(df, x="X", y="Y", color="Language",symbol="Language",text="Words")
    # adjusting the text position
    fig.update_traces(textposition='top center')
    # setting up the height and title
    fig.update_layout(
        title_text='Word embedding chart'
    )
    # displaying the figure
    pio.write_image(fig,"/Users/harsharenkila/PycharmProjects/ANLP/project/output/word_visualization_plots/{}_words_viz.png".format(model_name))
    fig.show()


def plot_bar_graph(langs,cluster_values,model_name,column,clustering_model):
    fig = plt.figure()
    plt.bar(langs, cluster_values,color ='maroon',
        width = 0.4)
    plt.xlabel("Langauage pairs")
    plt.ylabel("No. of data points have same cluster")
    plt.title("{} {} Multi cluster analysis".format(model_name,column))
    if clustering_model == None:
        clustering_model = "GMM"
    else:
        clustering_model = "K_Means"
    plt.savefig("/Users/harsharenkila/PycharmProjects/ANLP/project/output/cluster_plots/{}_{}_{}_cluster_plot.png".format(model_name,column,clustering_model))
    #plt.show()
