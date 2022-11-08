import fasttext
import pandas as pd

ft = fasttext.load_model('/Users/harsharenkila/PycharmProjects/ANLP/project/fasttext_models/cc.en.300.bin')

ft.get_dimension()

print(ft.get_word_vector('hello').shape)
languages_list = [("dutch","NL"),("spanish","ES"),("german","DE"),("french","FR")]
path = "/Users/harsharenkila/PycharmProjects/ANLP/project/data/EN-{}.csv"
def embeddings(language,code,file_path,model_name):
    df = pd.read_csv(file_path.format(code))
    print(len(df["{}_title".format(language)].tolist()))
    #title = df["{}_title".format(language)].tolist()
    embeddings = []
    for i in range(len(df)):
        embeddings.append(ft.get_word_vector(df["{}_title".format(language)].iloc[i]))
    df["{}_title_embeddings".format(language)] = embeddings
    print("title_completed")
    del embeddings
    description_embeddings = []
    for i in range(len(df)):
        description_embeddings.append(ft.get_word_vector(df["{}_description".format(language)].iloc[i]))
    df["{}_description_embeddings".format(language)] = description_embeddings
    print("description complted")
    del description_embeddings
    domain_embeddings = []
    for i in range(len(df)):
        domain_embeddings.append(ft.get_word_vector(df["{}_domain".format(language)].iloc[i]))
    df["{}_domain_embeddings".format(language)] = domain_embeddings
    del domain_embeddings
    df.to_csv("/Users/harsharenkila/PycharmProjects/ANLP/project/embeddings/{}_{}_data.csv".format(model_name,language))


'''models = [("/Users/harsharenkila/PycharmProjects/ANLP/project/fasttext_models/cc.es.300.bin","spanish","ES","FastText"),("/Users/harsharenkila/PycharmProjects/ANLP/project/fasttext_models/cc.fr.300.bin","french","FR","FastText"),("/Users/harsharenkila/PycharmProjects/ANLP/project/fasttext_models/cc.de.300.bin","german","DE","FastText"),("/Users/harsharenkila/PycharmProjects/ANLP/project/fasttext_models/cc.nl.300.bin","dutch","NL","FastText")]
for mod in models:
    ft = fasttext.load_model(mod[0])
    print(mod)

    embeddings(mod[1],mod[2],path,mod[3])'''
#for lan in languages_list:
#    df = pd.read_csv(path.format(lan[1]))
