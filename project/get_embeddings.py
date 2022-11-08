from sentence_transformers import SentenceTransformer
import pandas as pd

#https://fasttext.cc/docs/en/crawl-vectors.html
path = "project/data/EN-{}.csv"
#model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
def embeddings(language,code,file_path,model_name):
    df = pd.read_csv(file_path.format(code))
    print(len(df["{}_title".format(language)].tolist()))
    df["{}_title_embeddings".format(language)] = model.encode(df["{}_title".format(language)].tolist()).tolist()
    print("title_completed")
    df["{}_description_embeddings".format(language)] = model.encode(df["{}_description".format(language)].tolist()).tolist()
    print("description complted")
    df["{}_domain_embeddings".format(language)] = model.encode(df["{}_domain".format(language)].tolist()).tolist()
    df.to_csv("project/embeddings/{}_{}_data.csv".format(model_name,language))

languages_list = [("dutch","NL"),("spanish","ES"),("german","DE"),("french","FR")]

models = [("google/mt5-large","M_T5")]
for mod in models:
    model = SentenceTransformer(mod[0],device="gpu")
    print(mod)
    for lan in languages_list:
        print(lan)
        embeddings(lan[0],lan[1],path,mod[1])
#embeddings("dutch","NL",path)

