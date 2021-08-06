import pandas as pd 
import string 
import json
import math
import numpy as np
import time
import re

class Content():

    def __init__(self):
        self.unique_words = set()

    def read_ratings(self, ratings_path):
        ''' Essa funcao faz um pre-processamento do CSV que contem os ratings'''

        df = pd.read_csv(ratings_path)
        self.user_ratings = {}

        for row in df.itertuples():
            userid = row[1].split(":")[0]
            itemid = row[1].split(":")[1]
            # atribui identificadores aos usuarios
            if userid not in self.user_ratings.keys():
                self.user_ratings[userid] = {}
            self.user_ratings[userid][itemid] = int(row.Prediction)

    def preprocess_text(self, text):
        text = re.sub(r'\d+', '', text) # remove numeros 
        text = re.sub('([A-Z]{1})', r'_\1', text).lower() # deixar tudo minusculo

        s = list(text)
        for i, t in enumerate(s):
            if t == "'" and i < len(s) - 1:
                s[i] = ' '
                if s[i+1].isalpha():
                    s[i+1] = ''
            elif t in string.punctuation:  # remove pontuacoes
                s[i] = ' '

        text = "".join(s)
        text = text.replace('  ', ' ')
        
        return text


    def tokenize(self, text, sep):
        stopwords_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
                        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
                        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
                        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
                        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
                        'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
                        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
                        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                        'won', "won't", 'wouldn', "wouldn't"]

        tokens = text.split(sep)
        # print(tokens)
        tokens = [token for token in tokens if token not in list(string.punctuation)+stopwords_en and token != '']
        # print(tokens)
        return tokens


    def preprocessing(self, content, itemid):
        content_tokenized = {}
        
        del_features = ["Year", "Rated", "Released", "Runtime", "Awards", "Poster", "Metascore", "imdbRating", "imdbVotes", "imdbID", "Type", "Response"]

        for key in content.keys():
            description = content[key]
            if key == "imdbRating":
                # print(content[key])
                # print(key)
                # if itemid == "i0414387":
                #     print(content[key])
                self.imdbRating[itemid] = content[key]
                

            if description != 'N/A': # and key == "Plot":
                # print(content[key])
                description = self.preprocess_text(content[key])
                # print(description)
                if key == 'Director' or key == 'Actors' or key == 'Writer':  # como tratar os dados de nomes
                    if '(' in description:
                        flag_erase = 0
                        s = list(description)
                        for i in range(len(description)):
                            if description[i] == '(':
                                flag_erase = 1
                            if flag_erase == 1:
                                s[i] = ''
                            if description[i] == ')':
                                flag_erase = 0
                        description = "".join(s)
                    tokens = self.tokenize(description.replace(' ', ''), sep=',')
                else:
                    tokens = self.tokenize(description, sep=' ')
                # res = []
                # for token in tokens:
                #     if token not in stopwords_en:
                #         # token = self.check_capital_letters(token)
                #         if token not in stopwords_en:
                #             res.append(token)
                self.unique_words.update(tokens)
                    
                content_tokenized[key] = tokens
            
        content = []
        # print(content_tokenized.keys())
        for key in content_tokenized.keys():
            if key not in del_features:
                content.extend(content_tokenized[key])
        

        return content

    def count_occurencies(self, tokens):
        tokenCount = dict.fromkeys(self.unique_words, 0) 
        for token in tokens:
            if token in self.unique_words:
                tokenCount[token] += 1
        
        return tokenCount

    def document_frequency(self):
        # number of documents that the word appears
        docFreq = dict.fromkeys(self.unique_words, 0)
        for item, tokens in self.contents.items():
            for w in np.unique(tokens):
                if w in self.unique_words:
                    docFreq[w] += 1
        return docFreq

    # def term_frequency(self, tokens):
    #     tf = {}
    #     tokenCount = self.count_occurencies(tokens)
    #     for token, count in tokenCount.items():
    #         try:
    #             tf[token] += tokenCount[token]/len(tokens)
    #         except:
    #             tf[token] = tokenCount[token]/len(tokens)
    #     return tf
    
    def compute_idf(self):
        start = time.time()
        docFreq = self.document_frequency()
        end = time.time() - start
        # print("Time compute document frequency")
        # print(end)
        # print("Document Frequency\n")
        # print(docFreq)
        idf = {}
        N = len(self.contents)
        for token, count in docFreq.items():
            idf[token] = math.log10(N/count)
        return idf

    def filtering_words(self):
        docFreq = self.document_frequency()
        for token, count in docFreq.items():
            if count < 0.005*len(self.contents):
                self.unique_words.remove(token)

    def compute_tf_idf(self):
        start = time.time()
        idf = self.compute_idf()
        end = time.time() - start
        # print("Time compute idf")
        # print(end)
        # print("IDF\n")
        # print(idf)
        tf_idf = {}
        start = time.time()
        for item, tokens in self.contents.items():
            tf_idf[item] = dict.fromkeys(self.unique_words, 0) #[0] * len(self.unique_words)
            if tokens == []:
                continue

            tokenCount = self.count_occurencies(tokens)
            for token, count in tokenCount.items():
                tf = tokenCount[token]/len(tokens) # computa o tf
                tf_idf[item][token] = tf * idf[token]
        #     print(item)
        # end = time.time() - start
        # print("Time compute tf-idf")
        # print(end)
        return tf_idf

    def read_content(self, contents_path):
        self.contents = {}
        self.imdbRating = {}
        i = 0
        with open(contents_path, 'r') as f:
            
            for content in f:
                if i == 0:
                    i+= 1
                    continue
                itemid = content.split(',')[0]
                self.contents[itemid] = self.preprocessing(json.loads(content[9:]), itemid)
                
        # print(self.unique_words)
        # print(len(self.unique_words))
        self.filtering_words()
        # print(len(self.unique_words))
        self.item_vectors = self.compute_tf_idf()

    
    def cosine_similarity(self, user, item):
        return np.dot(user, item)/(np.linalg.norm(user) * np.linalg.norm(item))


    def compute_user_vectors(self):
        self.user_vectors = {}
        for userid, items in self.user_ratings.items():
            self.user_vectors[userid] = np.zeros(len(self.unique_words))
            for itemid, rating in items.items():
                # print(self.item_vectors[itemid].values())
                # print(type(self.item_vectors[itemid].values()))
                # print(len(self.user_vectors[userid]))
                self.user_vectors[userid] += np.array(list(self.item_vectors[itemid].values()))*rating
            self.user_vectors[userid] /= len(items)

    def prediction(self, userid, itemid):
        return self.cosine_similarity(self.user_vectors[userid], np.array(list(self.item_vectors[itemid].values())))*10

    
    def submission(self, targets_path):
        ''' Essa funcao itera pelas tuplas de usuarios e itens disponiveis no targets.csv
            e realiza a predicao para cada uma delas '''

        df = pd.read_csv(targets_path)

        print("UserId:ItemId,Prediction")

        for row in df.itertuples():
            userid = row[1].split(":")[0]
            itemid = row[1].split(":")[1]

            if userid in self.user_ratings.keys():
                if np.linalg.norm(np.array(list(self.item_vectors[itemid].values()))) == 0:
                    print(itemid)
                    break
                prediction = self.prediction(userid, itemid)

                # se a predicao extrapolar a menor ou a maior nota da escala, substitui a nota
                if prediction > 10:
                    prediction = 10
                elif prediction < 0:
                    prediction = 0

            # caso de cold-start de item
            # elif userid in self.user_index.keys():
            #     prediction = self.b_u[self.user_index[userid]] + self.b 

            # # caso de cold-start de usuario
            # elif itemid in self.item_index.keys():
            #     prediction = self.b_i[self.item_index[itemid]] + self.b 
            
            # cold-start tanto de usuario quanto de item
            else:
                # print(itemid)
                # print(self.imdbRating)
                prediction = self.imdbRating[itemid]

            print("{}:{},{}".format(userid, itemid, prediction))