import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow_hub as hub

from project import *
from project.src import data_preprocessing
from project.src.evaluation import cluster_distributional_analysis
from project.src.load_embeddings import Vectors
from project.src.model_building import Model_Building
from project.src.training_and_evaluation import train
from project.utils.data_operations import get_data_from_file
from project.utils.plotting import visualize_word_vectors, get_sample_index

print("-------Data preprocessing stated------")

#data_preprocessing.preprocess_data("project/data/Occupation Data.xlsx","project/data/Job Zones.xlsx")

print("----- New file created after preprocessing --------" )


print("Visualize word embeddings in 2-D space")
index =  get_sample_index()
visualize_word_vectors("Multi_Bert","title",index)
visualize_word_vectors("Xlmr_Bert","title",index)
visualize_word_vectors("M_USE","title",index)
visualize_word_vectors("MDistill","title",index)
#visualize_word_vectors("Multi_Bert","description",index)
#visualize_word_vectors("Xlmr_Bert","description",index)
#visualize_word_vectors("M_USE","description",index)
#visualize_word_vectors("MDistill","description",index)


print("-------Started Multi cluster analysis of Embeddings----------")
#### Multi cluster distribution Analysis

'''for model in models:
    title = []
    description =[]
    for lang in langauges:
        output = get_data_from_file(model=model,language=lang)
        try:
            if(output == False):
                print("error")
            t,des,do,job_zone = output
        except:
            t,des,do,job_zone = output
        title.append(t)
        description.append(des)

    cluster_distributional_analysis(title,10,model,"Title","spectral")
    cluster_distributional_analysis(description,10,model,"Description","spectral")

    cluster_distributional_analysis(title,10,model,"Title")
    cluster_distributional_analysis(description,10,model,"Description")'''

print("-------completed------")
'''embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
en_result = embed(english_sentences)
print(en_result)'''
#model = Vectors("sadakmed/distiluse-base-multilingual-cased-v1","harsha the great")
#model.load_sentence_transformer_model()
#print(model.encode_text())
'''print("----------------Model Architecture defining---------------- ")
models = Model_Building()
usemodel = models.USEmodel()


output = get_data_from_file()
try:
    if(output == False):
        print("error")
    t,des,do,job_zone = output
except:
    t,des,do,job_zone = output

print(t)
print(type(t))
print(len(output))
print(t.shape,des.shape,do.shape)
print(job_zone)
job_zone = [x - 1 for x in job_zone]

d = {x:job_zone.count(x) for x in job_zone}
print(d)
Y = to_categorical(job_zone,num_classes=5)
print(Y.shape)
print(Y)

train("use_english",usemodel,t,des,do,Y,100,1e-3)'''

