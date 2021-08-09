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
        self.features = ['Plot', 'Genre', 'Director', 'Year', 'Actors', 'Writer', 'Title']

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
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower()

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
        tokens = [token for token in tokens if token not in list(string.punctuation)+stopwords_en and token != '']
   
        return tokens


    def preprocessing(self, content, itemid):
        content_tokenized = {}
        
        #del_features = ['seriesID', 'Writer', 'Runtime', 'Language', 'Metascore', 'Title', 'Poster', 'Season', 
                    #'Director', 'imdbID', 'Response', 'Genre', 'imdbVotes', 'Episode', 'Year', 'Rated', 
                    # 'Plot', 'Released', 'Country', 'Actors', 'imdbRating', 'Type', 'Awards', 'Error']
        del_features = ['seriesID', 'Writer', 'Runtime', 'Language', 'Metascore', 'Title', 'Poster', 'Season', 
                        'Director', 'imdbID', 'Response', 'imdbVotes', 'Episode', 'Year', 'Rated', 'Genre',
                        'Released', 'Country', 'Actors', 'imdbRating', 'Type', 'Awards', 'Error']
        for key in self.features:
            if key not in content.keys() or content[key] == 'N/A':
                content_tokenized[key] = []
                continue
            if key == 'Director' or key == 'Actors' or key == 'Writer':
                text = content[key]
                if '(' in text:
                    flag_erase = 0
                    s = list(text)
                    for i in range(len(text)):
                        if text[i] == '(':
                            flag_erase = 1
                        if flag_erase == 1:
                            s[i] = ''
                        if text[i] == ')':
                            flag_erase = 0
                    text = "".join(s)
                # preprocessed_text = self.preprocess_text(text)
                content_tokenized[key] = self.tokenize(text.lower(), sep=',')
           
            elif key == 'Year':
                year = content[key][:3]
                content_tokenized[key] = [year]
                # if int(content[key][3:]) >= 5:
                #     content_tokenized[key] = [str(int(year)+1)]
                # if int(content[key][3:]) < 5:
                #     content_tokenized[key] = [str(int(year)-1)]
                # print(content_tokenized['Year'])
                self.unique_words.add(year)

            else: 
                description = self.preprocess_text(content[key])
                tokens = self.tokenize(description, sep=' ')
                content_tokenized[key] = tokens
                # self.unique_words.update(tokens)

        if 'imdbRating' in content.keys() and content['imdbRating'] != 'N/A':
            self.imdbRating[itemid] = float(content['imdbRating'])
            self.imdbAvgRating += float(content['imdbRating'])

        return content_tokenized

    def count_occurencies(self, tokens):
        tokenCount = {} #dict.fromkeys(np.unique(tokens), 0) 
        for token in tokens:
            try: 
                tokenCount[token] += 1
            except:
                tokenCount[token] = 1
        
        return tokenCount

    def document_frequency(self, feature):
        # number of documents that the word appears
        docFreq = {}
        for item, values in self.contents.items():
            tokens = self.contents[item][feature]
            for w in np.unique(tokens):
                try:
                    docFreq[w] += 1
                except:
                    docFreq[w] = 1
        return docFreq
    
    def compute_idf(self, feature):
        start = time.time()
        docFreq = self.document_frequency(feature)
        end = time.time() - start

        idf = {}
        N = len(self.contents)
        for token, count in docFreq.items():
            idf[token] = np.log(N/count)
        return idf


    def compute_tf_idf(self, feature):
        start = time.time()
        idf = self.compute_idf(feature)
        end = time.time() - start
        tf_idf = {}
        start = time.time()
        tf_idf_norm = {}
        for item, values in self.contents.items():
            tokens = self.contents[item][feature]
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
        self.item_vectors = {}
        self.item_norms = {}
        self.item_vectors['Plot'], self.item_norms['Plot'] = self.compute_tf_idf('Plot')
        self.item_vectors['Genre'], self.item_norms['Genre'] = self.compute_tf_idf('Genre')
        self.item_vectors['Year'], self.item_norms['Year'] = self.compute_tf_idf('Year')
        # print(self.contents)
        self.item_vectors['Director'], self.item_norms['Director'] = self.compute_tf_idf('Director')
        self.item_vectors['Actors'], self.item_norms['Actors'] = self.compute_tf_idf('Actors')
        self.item_vectors['Writer'], self.item_norms['Writer'] = self.compute_tf_idf('Writer')
        self.item_vectors['Title'], self.item_norms['Title'] = self.compute_tf_idf('Title')
        # self.item_vectors['Country'], self.item_norms['Country'] = self.compute_tf_idf('Country')



    def cosine_similarity(self, id1, id2, feature):
        sim = 0
        for term in self.item_vectors[feature][id1].keys():
            if term in self.item_vectors[feature][id2]:
                sim += self.item_vectors[feature][id1][term] * self.item_vectors[feature][id2][term] 

        if sim == 0:
            # print(id1)
            return 0
        
        return sim/(self.item_norms[feature][id1] * self.item_norms[feature][id2])

    def prediction(self, userid, itemid, feature):
        sim_sum = 0
        numerador = 0
        # print(self.user_ratings[userid])
        
        for item, rating in self.user_ratings[userid].items():
            if self.item_norms[feature][item] != 0:
                sim = self.cosine_similarity(item, itemid, feature)
                sim_sum += sim
                numerador += sim*rating

        if sim_sum == 0:
            # print(itemid)
            return 0
        
        return numerador/sim_sum

    def input_avg_rating_user(self, userid, itemid):
        prediction = 0
        sum_weights = 0
        for item, rating in self.user_ratings[userid].items():
            prediction += rating
        prediction /= len(self.user_ratings[userid])

        if itemid in self.imdbRating.keys():
            prediction += self.imdbRating[itemid]
            prediction /= 2
        
        return prediction

    def aggregate_predictions(self, predictions):
        # ['Plot', 'Genre', 'Director', 'Year', 'Actors', 'Writer']
        # ['Plot', 'Genre', 'Director', 'Year', 'Actors', 'Writer', 'Title']
        weights = [4, 3, 4, 5, 3, 3, 4] 
        weighted_sum = 0
        for i, pred in enumerate(predictions):
            weighted_sum += weights[i]*pred
        
        return weighted_sum/np.sum(weights)

    def submission(self, targets_path):
        ''' Essa funcao itera pelas tuplas de usuarios e itens disponiveis no targets.csv
            e realiza a predicao para cada uma delas '''

        df = pd.read_csv(targets_path)

        print("UserId:ItemId,Prediction")

        for row in df.itertuples():
            userid = row[1].split(":")[0]
            itemid = row[1].split(":")[1]
            predictions = []
            if userid in self.user_ratings.keys():
                for feature in self.features:
                    if self.item_norms[feature][itemid] != 0:
                        pred = self.prediction(userid, itemid, feature)
                        if pred != 0:
                            predictions.append(pred)
                    if pred == 0 or self.item_norms[feature][itemid] == 0:
                        predictions.append(self.input_avg_rating_user(userid, itemid))
            
                prediction = self.aggregate_predictions(predictions)

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
        