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
        # text = re.sub(r'\d+', '', text) # remove numeros 
        # text = re.sub('([A-Z]{1})', r'\1', text).lower() # deixar tudo minusculo
        # text = re.sub(u"[àáâãäå]", 'a', text)
        # text = re.sub(u"[èéêë]", 'e', text)
        # text = re.sub(u"[ìíîï]", 'i', text)
        # text = re.sub(u"[òóôõö]", 'o', text)
        # text = re.sub(u"[ùúûü]", 'u', text)
        # s = list(text)
        # for i, t in enumerate(s):
        #     # if t == "'" and i < len(s) - 1:
        #     #     s[i] = ' '
        #     #     if s[i+1].isalpha():
        #     #         s[i+1] = ''
        #     # elif t == '-':
        #     #     continue
        #     if t in string.punctuation:  # remove pontuacoes
        #         s[i] = ''

        # text = "".join(s)
        # text = text.replace('  ', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower()
        
        # return text


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
        
        #del_features = ['seriesID', 'Writer', 'Runtime', 'Language', 'Metascore', 'Title', 'Poster', 'Season', 
                    #'Director', 'imdbID', 'Response', 'Genre', 'imdbVotes', 'Episode', 'Year', 'Rated', 
                    # 'Plot', 'Released', 'Country', 'Actors', 'imdbRating', 'Type', 'Awards', 'Error']
        del_features = ['seriesID', 'Writer', 'Runtime', 'Language', 'Metascore', 'Title', 'Poster', 'Season', 
                        'Director', 'imdbID', 'Response', 'imdbVotes', 'Episode', 'Year', 'Rated', 'Genre',
                        'Released', 'Country', 'Actors', 'imdbRating', 'Type', 'Awards', 'Error']

        content_tokenized['Plot'] = []
        if 'Plot' in content.keys() and content['Plot'] != 'N/A':
            # if key == 'Director' or key == 'Actors' or key == 'Writer':  # como tratar os dados de nomes
            # description = content['Director']
            # if '(' in description:
            #     flag_erase = 0
            #     s = list(description)
            #     for i in range(len(description)):
            #         if description[i] == '(':
            #             flag_erase = 1
            #         if flag_erase == 1:
            #             s[i] = ''
            #         if description[i] == ')':
            #             flag_erase = 0
            #     description = "".join(s)
            description = self.preprocess_text(content['Plot'])
            tokens = self.tokenize(description, sep=' ')
            content_tokenized['Plot'] = tokens
            self.unique_words.update(tokens)

        if 'imdbRating' in content.keys() and content['imdbRating'] != 'N/A':
            self.imdbRating[itemid] = float(content['imdbRating'])
            self.imdbAvgRating += float(content['imdbRating'])

        return content_tokenized['Plot']

    def count_occurencies(self, tokens):
        tokenCount = {} #dict.fromkeys(np.unique(tokens), 0) 
        for token in tokens:
            try: 
                tokenCount[token] += 1
            except:
                tokenCount[token] = 1
        
        return tokenCount

    def document_frequency(self):
        # number of documents that the word appears
        docFreq = {}
        for item, tokens in self.contents.items():
            for w in np.unique(tokens):
                try:
                    docFreq[w] += 1
                except:
                    docFreq[w] = 1
        return docFreq
    
    def compute_idf(self):
        start = time.time()
        docFreq = self.document_frequency()
        end = time.time() - start

        idf = {}
        N = len(self.contents)
        for token, count in docFreq.items():
            idf[token] = np.log(N/count)
        return idf


    def compute_tf_idf(self):
        start = time.time()
        idf = self.compute_idf()
        end = time.time() - start
        tf_idf = {}
        start = time.time()
        tf_idf_norm = {}
        for item, tokens in self.contents.items():
            tf_idf[item] = {} 
            tf_idf_norm[item] = 0
            if tokens == []:
                continue

            tokenCount = self.count_occurencies(tokens)
            for token, count in tokenCount.items():
                tf = tokenCount[token]/len(tokens) # computa o tf
                tf_idf[item][token] = tf * idf[token]
                tf_idf_norm[item] += tf_idf[item][token]*tf_idf[item][token]

            tf_idf_norm[item] = math.sqrt(tf_idf_norm[item])

        return tf_idf, tf_idf_norm

    def read_content(self, contents_path):
        self.contents = {}
        self.imdbRating = {}
        self.imdbAvgRating = 0
        self.gendersAvgRating = {}
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
        self.item_vectors, self.item_norms = self.compute_tf_idf()

    
    def cosine_similarity(self, id1, id2):
        sim = 0
        for term in self.item_vectors[id1].keys():
            if term in self.item_vectors[id2]:
                sim += self.item_vectors[id1][term] * self.item_vectors[id2][term] 

        if sim == 0:
            # print(id1)
            return 0
        
        return sim/(self.item_norms[id1] * self.item_norms[id2])

    def prediction(self, userid, itemid):
        sim_sum = 0
        numerador = 0
        # print(self.user_ratings[userid])
        
        for item, rating in self.user_ratings[userid].items():
            if self.item_norms[item] != 0:
                sim = self.cosine_similarity(item, itemid)
                sim_sum += sim
                numerador += sim*rating

        if sim_sum == 0:
            # print(itemid)
            return 0
        
        return numerador/sim_sum

    def input_avg_rating_user(self, userid):
        prediction = 0
        sum_weights = 0
        for item, rating in self.user_ratings[userid].items():
            # if item in self.imdbRating.keys():
            #     sum_weights += self.imdbRating[item]
            #     prediction += rating*self.imdbRating[item]
            # else:
                sum_weights += 1
                prediction += rating
        prediction /= sum_weights #len(self.user_ratings[userid])
        return prediction

    def submission(self, targets_path):
        ''' Essa funcao itera pelas tuplas de usuarios e itens disponiveis no targets.csv
            e realiza a predicao para cada uma delas '''

        df = pd.read_csv(targets_path)

        print("UserId:ItemId,Prediction")

        for row in df.itertuples():
            userid = row[1].split(":")[0]
            itemid = row[1].split(":")[1]

            if userid in self.user_ratings.keys():
                if self.item_norms[itemid] != 0:
                    prediction = self.prediction(userid, itemid)
                if prediction == 0 or self.item_norms[itemid] == 0:
                    prediction = self.input_avg_rating_user(userid)
         
                if prediction > 10:
                    prediction = 10
                elif prediction < 0:
                    prediction = 0

            # cold-start de usuario
            elif itemid in self.imdbRating.keys():
                prediction = self.imdbRating[itemid]
            else:
                prediction = self.imdbAvgRating/len(self.imdbRating)

            print("{}:{},{}".format(userid, itemid, prediction))