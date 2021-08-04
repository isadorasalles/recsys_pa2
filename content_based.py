import pandas as pd 
import string 

class Content():
    def read_ratings(self, ratings_path):
        ''' Essa funcao faz um pre-processamento do CSV que contem os ratings'''

        df = pd.read_csv(ratings_path)
        self.user_index = {}
        self.item_index = {}

        u_ind = 0
        i_ind = 0
        self.ratings = []

        for row in df.itertuples():
            userid = row[1].split(":")[0]
            itemid = row[1].split(":")[1]
            # atribui identificadores aos usuarios
            if userid not in self.user_index.keys():
                self.user_index[userid] = u_ind
                u_ind+=1
            # atribui identificadores aos itens
            if itemid not in self.item_index.keys():
                self.item_index[itemid] = i_ind
                i_ind+=1
            # armazena lista de tuplas para facilitar o acesso
            self.ratings.append((self.user_index[userid], self.item_index[itemid], int(row.Prediction)))

        self.b = np.mean(df["Prediction"]) # b representa o bias global, que eh dado pela media das avaliacoes
        self.len_item = i_ind
        self.len_user = u_ind

    def check_capital_letters(self, token):
        s = list(token)
        for i, t in enumerate(s):
            # print(t)
            if t in string.punctuation:
                s[i] = ''
                continue
            if ord(t) >= 65 and ord(t) <= 90:
                s[i] = chr(ord(t)+32)
        # print(s)
        token = "".join(s)
        return token

    def preprocessing(self, content):
        content = content.replace('{', '').replace('}', '')
        features = content.split('","')
        content_splitted = {}
        stopwords_en = [u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', \
                        u"she's", u'her', u'hers', u'herself', u'it', u"it's", u'its', u'itself', u'they', u'them', \
                        u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', \
                        u'that', u"that'll", u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', \
                        u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', \
                        u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', \
                        u'while', u'of', u'at'] 
        del_features = ["Poster", "Metascore", "imdbRating", "imdbVotes", "imdbID", "Type", "Response"]
        # punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\', ']', '_', '|','}', '~']
        # ponctuation
        for f in features:
            #  print(f)
            description = f.split(':')[1]
            type_ = f.split(':')[0]

            if type_.replace('"', '') in del_features:
                continue
            # print(description)
            if description != '"N/A':
                # print(type_)
                if type_ == 'Director"' or type_ == 'Actors"' or type_ == 'Writer"':
                    # print(description)
                    tokens = description.replace('"', '').replace(' ', '').split(',')
                else:
                    tokens = description.replace('"', '').split(' ')
                res = []
                for token in tokens:
                    if token not in stopwords_en:
                        token = self.check_capital_letters(token)
                        if token not in stopwords_en:
                            res.append(token)
                    
                content_splitted[type_.replace('"', '')] = res
            
        content = []
        for key in content_splitted.keys():
            content.extend(content_splitted[key])
        return content
    
    # tirar os n/a - FEITO
    # excluir colunas de poster, matascore, imdbrating, imdbvotes, imdbID, type e response - FEITO
    # nas colunas de diretor, escritor e atores podemos splitar por virgulas e juntar os nomes e sobrenomes - FEITO
    # excluir pontuações? - FEITO - verificar se pode usar essa bib

    def read_content(self, contents_path):
        self.contents = {}
        i = 0
        with open(contents_path, 'r') as f:
            
            for content in f:
                if i == 0:
                    i+= 1
                    continue
                # print(content)
                self.contents[content.split(',')[0]] = self.preprocessing(content[9:])
                # print(self.contents)
                # break
            # print(self.contents['i4967094'])
