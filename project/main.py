import pandas as pd

from project import data_preprocessing
from project import *
from project.src.model_building import Model_Building
from project.utils.data_operations import get_data_from_file
from keras.utils import to_categorical
from project.src.training_and_evaluation import train
from project.utils.plotting import visualize_word_vectors

print("-------Data preprocessing stated------")

#data_preprocessing.preprocess_data("project/data/Occupation Data.xlsx","project/data/Job Zones.xlsx")

print("----- New file created after preprocessing --------" )

print("Visualize word embeddings in 2-D space")
df = pd.read_csv("/Users/harsharenkila/PycharmProjects/ANLP/project/embeddings/FastText_dutch_data.csv")
index = df.sample(5, replace=False).index
del df
visualize_word_vectors("Multi_Bert","title",index)
visualize_word_vectors("Xlmr_Bert","title",index)
visualize_word_vectors("MUSE","title",index)

print("----------------Model Architecture defining---------------- ")
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

train("use_english",usemodel,t,des,do,Y,100,1e-3)