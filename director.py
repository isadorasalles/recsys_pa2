        # for key in content.keys():
        #     description = content[key]

        #     if description != 'N/A': # and key == "Plot":
        #         if key == "imdbRating":
        #             self.imdbRating[itemid] = content[key]
        #             self.imdbAvgRating += float(content[key])
        #         # print(content[key])
        #         # print(description)
        #         if key == 'Director' or key == 'Actors' or key == 'Writer':  # como tratar os dados de nomes
        #             if '(' in description:
        #                 flag_erase = 0
        #                 s = list(description)
        #                 for i in range(len(description)):
        #                     if description[i] == '(':
        #                         flag_erase = 1
        #                     if flag_erase == 1:
        #                         s[i] = ''
        #                     if description[i] == ')':
        #                         flag_erase = 0
        #                 description = "".join(s)
        #             description = re.sub('([A-Z]{1})', r'_\1', description).lower()
        #             tokens = self.tokenize(description, sep=',')
        #             tokens = [self.preprocess_text(token).replace(' ', '') for token in tokens]
        #         else:
        #             description = self.preprocess_text(content[key])
        #             tokens = self.tokenize(description, sep=' ')

        #         if key not in del_features:
        #             self.unique_words.update(tokens)

        #         if key == 'Genre':
        #             for token in tokens:
        #                 if content['imdbRating'] != 'N/A':
        #                     try:
        #                         self.gendersAvgRating[token].append(float(content['imdbRating']))
        #                     except:
        #                         self.gendersAvgRating[token] = []
        #                         self.gendersAvgRating[token].append(float(content['imdbRating']))
                    
        #         content_tokenized[key] = tokens
            
        # content = []
        # # print(content_tokenized.keys())
        # for key in content_tokenized.keys():
        #     if key not in del_features:
        #         content.extend(content_tokenized[key])
        
        # for key in self.gendersAvgRating.keys():
        #     self.gendersAvgRating[key] = np.mean(self.gendersAvgRating[key])


            # def compute_user_vectors(self):
    #     self.user_vectors = {}
    #     for userid, items in self.user_ratings.items():
    #         self.user_vectors[userid] = {} #np.zeros(len(self.unique_words))
    #         for itemid, rating in items.items():
    #             # print(self.item_vectors[itemid].values())
    #             # print(type(self.item_vectors[itemid].values()))
    #             # print(len(self.user_vectors[userid]))
    #             for term, value in self.item_vectors[itemid].items():
    #                 try: 
    #                     self.user_vectors[userid][term] += (value*rating)/len(items)
    #                 except:
    #                     self.user_vectors[userid][term] = (value*rating)/len(items)
    #             # self.user_vectors[userid] += np.array(list(self.item_vectors[itemid].values()))*rating
    #         # self.user_vectors[userid] /= len(items)