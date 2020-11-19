import csv
import json
import os.path
from os import path
import numpy as np

import pprint

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


# P(tag|word) = P(word|tag) * P(tag)/P(word)

# NN - (1/12,3/12,5/12,2/12,1/12)
# PSP - (1,3,4,2,1)

# word2 - (3/12, 3/11, ....)

# tag_dict: <word: max_tag_freq> {"word": {"tag1": freq ... }}
# generate dict
# get_word

class Tagger:

    __train_file_name = "train_set.csv"
    __test_file_name = "test_set.csv"
    __tag_dict_file_name = "model.json"
    __tags_frequency_file_name = "tags_frequency.csv"
    __words_frequency_file_name = "words_frequency.csv"
    

    # {"word": {"tag1": freq ... }}
    tag_dict = None

    tags_list = None
    tags_prob_list = None

    words_list = None
    words_prob_list = None

    word_freq = None
    word_count = 0

    y_predicted = []
    y_actual = []

    no_of_unique_tags = None
    no_of_unique_words = None

    # [tags][words]
    p_word_tag = None

    p_word = None

    p_tag = None

    def __init__(self):
        self.generate_dict()

    def check_dict(self):
        if (path.exists(self.__tag_dict_file_name) == True):
            with open(self.__tag_dict_file_name, "r") as jsonFile:
                content = jsonFile.read()
                self.tag_dict = json.loads(content)
        else:
            self.generate_dict()

    def generate_word_tag(self):

        if(self.tag_dict is None):
            self.generate_dict()

        if(self.tags_list):
            self.generate_tags_data()
        
        self.no_of_unique_words = len(self.tag_dict)
        self.no_of_unique_tags = len(self.tags_list)

        word_tag_dict = {}

        # pp = pprint.PrettyPrinter()


        for word in self.tag_dict:
            tags = self.tag_dict[word]

            for tag in tags:
                words_map = word_tag_dict.get(tag)
                if (words_map is None):
                    words_map = {}
                if (words_map.get(word) is None):
                    words_map[word] = 0    
                words_map[word] = words_map[word] + tags[tag]  
                word_tag_dict[tag] = words_map

        # print(pp.pprint(word_tag_dict))

        for tag in word_tag_dict:
            words = word_tag_dict.get(tag)

            if(words != None):
                tot = np.sum(np.array(list(words.values())))
                for word in words:
                    word_tag_dict[tag][word] = word_tag_dict[tag][word] / tot

        self.p_word_tag = word_tag_dict            
                

    def generate_tags_data(self):
        with open(self.__tags_frequency_file_name, "r") as tagsFile:
            reader = csv.reader(tagsFile)
            vals = []
            probs = []
            for entry in reader:
                vals.append(entry[0])
                probs.append(entry[1])
            probs = np.array(probs).astype(np.float)
            total = np.sum(probs)
            self.tags_prob_list = probs / total
            self.tags_list = vals
            
            p_tag = {}
            
            for idx, tag in enumerate(self.tags_list):
                p_tag[tag] = self.tags_prob_list[idx]

            self.p_tag = p_tag    

    def generate_words_data(self):
        with open(self.__words_frequency_file_name, "r") as wordsFile:
            reader = csv.reader(wordsFile, delimiter='~')
            val = []
            prob = []
            idx = -1
            for entry in reader:
                idx = idx + 1
                e1 = entry[0]
                e2 = entry[1]
                val.append(e1)
                prob.append(int(e2.strip()))     

            prob = np.array(prob).astype(np.float)
            total = np.sum(prob)
            self.words_prob_list = prob / total
            self.words_list = val

            p_word = {}
            
            for idx, word in enumerate(self.words_list):
                p_word[word] = self.words_prob_list[idx]

            self.p_word = p_word
    
    def generate_dict(self):
        self.tag_dict = {}
        with open(self.__train_file_name) as train_file:
            reader = csv.reader(train_file)
            # word, tag
            for entry in reader:
                tags = self.tag_dict.get(entry[0])
                if tags is None:
                    tags = {}
                if tags.get(entry[1]) != None:
                    tags[entry[1]] = tags[entry[1]] + 1
                else:
                    tags[entry[1]] = 1
                self.tag_dict[entry[0]] = tags

        with open(self.__tag_dict_file_name, "w") as outfile:
            json.dump(self.tag_dict, outfile)

    """Gets most probable tag for a given word using Bayes Theorem
       P(tag|word) = P(word|tag) * P(tag) / P (word)

    Parameters
    -----------
    word: String
        Input word for tag prediction

    Returns
    ----------
    String
        If word present in vocabulary(trainset) returns tag
        If Word absent in vocabulary use Smoothing 
        Technique and Returns tag according to 
        probabilistic distribution of all the tags          
    """
    def get_tags_prob(self, word):

        if self.p_tag is None:
            self.generate_tags_data()
        
        if self.p_word is None:
            self.generate_words_data()
        
        if self.p_word_tag is None:
            self.generate_word_tag()

        p_tag_word = {}

        for tag in self.tags_list:

            if(self.p_word_tag[tag].get(word) is None):
                self.p_word_tag[tag][word] = 0
            p_tag_word[tag] = (self.p_word_tag[tag][word] * self.p_tag[tag]) / self.p_word[word]

        ans = max(p_tag_word, key=p_tag_word.get) 
        
        if p_tag_word[ans] == 0.0:
            if self.tags_list is None:
                    self.generate_tags_data()
            ans = np.random.choice(self.tags_list, 1, p=self.tags_prob_list)[0]

        return ans          
            
    """Gets probable tag for a given input word
    Parameters
    ----------
    word : String
        Input word for tag predection
    mode : String    
        max -> select the tag with maximum frequency
        weighted -> select tag according to probabilistic distribution 
    
    Returns
    ---------
    String
        If word present in dataset returns tag according to [mode]
        else returns tag according to 
        probabilistic distribution of all the tags
    """
    def get_best_tag(self, word, mode):
        tags = self.tag_dict.get(word)

        if(tags != None):
            keys = np.array(list(tags.keys()))
            vals = np.array(list(tags.values()))
            tot = np.sum(vals)
            prob = vals/tot

            if (mode == "max"):
                max_tag = max(tags, key=tags.get)
                return max_tag
            else:
                return np.random.choice(keys, 1, p=prob)[0]

        else:
            if self.tags_list is None:
                self.generate_tags_data()
            return np.random.choice(self.tags_list, 1, p=self.tags_prob_list)[0]    

    def evaluate(self, mode):
        preds = []
        actuals = []

        with open(self.__test_file_name) as testFile:
            reader = csv.reader(testFile)
            next(reader)
            for entry in reader:
                # prediction1 = self.get_best_tag(entry[0], mode)
                prediction = self.get_tags_prob(entry[0])
                actual = entry[1]
                preds.append(prediction)
                actuals.append(actual)
            self.y_predicted = preds
            self.y_actual = actuals
            accuracy = str(round(self.find_accuracy(), 5))
        print("Test {} accuracy: {}".format(mode, accuracy))

    def confusion_matrix(self) :
        return confusion_matrix(self.y_actual, self.y_predicted, labels=self.tags_list)
    
    def find_accuracy(self) :
        return accuracy_score(self.y_actual, self.y_predicted)

print("Running")
tagger = Tagger()
tagger.evaluate("max")
# tagger.evaluate("weighted")

# tagger.generate_word_tag()



