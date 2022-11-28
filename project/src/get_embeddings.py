from sentence_transformers import SentenceTransformer
import pandas as pd

#https://fasttext.cc/docs/en/crawl-vectors.html
path = "project/data/EN-{}.csv"
#model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
def embeddings(language,code,file_path,model_name):
    df = pd.read_csv(file_path.format(code))
    print(len(df["Title".format(language)].tolist()))
    df["{}_title_embeddings".format(language)] = model.encode(df["Title"].tolist()).tolist()
    print("title_completed")
    df["{}_description_embeddings".format(language)] = model.encode(df["Description"].tolist()).tolist()
    print("description completed")
    df["{}_domain_embeddings".format(language)] = model.encode(df["Domain"].tolist()).tolist()
    df.to_csv("project/embeddings/{}_{}_data.csv".format(model_name,language))

languages_list = [("english","NL")]

models = [("bert-base-multilingual-cased","Multi_Bert")]#,("distiluse-base-multilingual-cased-v1","MUSE"),("xlm-r-100langs-bert-base-nli-stsb-mean-tokens","Xlmr_Bert")]
for mod in models:
    model = SentenceTransformer(mod[0])
    print(mod)
    for lan in languages_list:
        print(lan)
        embeddings(lan[0],lan[1],path,mod[1])
#embeddings("dutch","NL",path)

