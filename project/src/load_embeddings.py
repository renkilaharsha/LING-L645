from sentence_transformers import SentenceTransformer
import fasttext

class Vectors:
    def __init__(self,model_name,sentences,language:str=None):
        self.model_name = model_name
        self.language = language
        self.sentences = sentences

    def load_sentence_transformer_model(self):
        self.model = SentenceTransformer(self.model_name)
        print(self.model)
    def encode_text(self):
        if self.language != None:
            if(isinstance(self.sentences,list)):
                embed_list  = []
                for sent in self.sentences:
                    embed_list.append(self.fasttext_model.get_word_vector(sent))
                return embed_list
            else:

                return self.fasttext_model.get_word_vector(self.sentences)
        else:

            return self.model.encode(sentences=self.sentences)


    def load_fastext_model(self):
        if(self.language == "english"):
            self.fasttext_model = fasttext.load_model('/project/fasttext_models/cc.en.300.bin')
        if (self.language == "spanish"):
            self.fasttext_model = fasttext.load_model('/project/fasttext_models/cc.es.300.bin')
        if (self.language == "dutch"):
            self.fasttext_model = fasttext.load_model('/project/fasttext_models/cc.nl.300.bin')
        if (self.language == "german"):
            self.fasttext_model = fasttext.load_model('/project/fasttext_models/cc.de.300.bin')
        if (self.language == "french"):
            self.fasttext_model = fasttext.load_model('/project/fasttext_models/cc.fr.300.bin')
