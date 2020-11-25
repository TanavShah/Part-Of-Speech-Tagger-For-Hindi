import csv
import json
import os.path
from os import path
import numpy as np

import pprint

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

class Tagger:
    
    __train_file_name = "train_set.csv"
    __test_file_name = "test_set.csv"
    __tag_dict_file_name = "model.json"
    __tags_frequency_file_name = "tags_frequency.csv"
    __words_frequency_file_name = "words_frequency.csv"
    __train_hmm_file_name = "train_set_hmm.csv"
    __test_hmm_file_name = "test_set_hmm.csv"

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

    def read_train_file(self):           
        sentences = []
        tags = []
        sentence = []
        tag = []
        with open(self.__train_hmm_file_name) as trainHmm:
            reader = csv.reader(trainHmm, delimiter='~')
            for row in reader:
                if row[0] == " ":
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence=[]
                    tag=[]
                else:
                    w = row[0]
                    t = row[1]
                    sentence.append(w)
                    tag.append(t)
        print (tags[0])
        sentences = sentences[1:]
        tags = tags[1:]
        assert len(sentences) == len(tags)
        return (sentences,tags)

    def generate_emmission_probabilities(self):

        if(self.tag_dict is None):
            self.generate_dict()

        if(self.tags_list):
            self.generate_tags_data()
        
        self.no_of_unique_words = len(self.tag_dict)
        self.no_of_unique_tags = len(self.tags_list)

        word_tag_dict = {}

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

        for tag in word_tag_dict:
            words = word_tag_dict.get(tag)

            if(words != None):
                tot = np.sum(np.array(list(words.values())))
                for word in words:
                    word_tag_dict[tag][word] = word_tag_dict[tag][word] / tot

        self.p_word_tag = word_tag_dict  

    def generate_transition_probabilities(self) :

        self.generate_tags_data()

        transition_matrix = {}
        labels = []
        labels.append("start")

        for j in range(22) :
            labels.append(self.tags_list[j])

        d1 = dict()

        i = 0
        with open(self.__train_hmm_file_name) as trainHmm:
            reader = csv.reader(trainHmm, delimiter='~')
            
            for row in reader :
                if(row[0] == "~" or row[0] == "") :
                    continue

                if(i == 0) :
                    i += 1
                    continue
                
                position = -1
                for j in range(len(labels)) :
                    if(row[1] == labels[j]) :
                        position = j
                        break

                d1[i - 1] = [row[0], position]
                i += 1
        
        for j in range(23) :
            transition_matrix[j] = {}
            for k in range(23) :
                transition_matrix[j][k] = 0

        for j in range(len(d1)) :

            if(j == 0 or d1[j - 1][0] == ".") :
                transition_matrix[0][d1[j][1]] += 1 
                continue
            
            transition_matrix[d1[j - 1][1]][d1[j][1]] += 1


        total = 0

        for j in range (23) :
            total = 0
            for k in range (23) :
                total += transition_matrix[j][k]
            for k in range (23) :
                transition_matrix[j][k] /= total

        return transition_matrix


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

tagger = Tagger()
# tagger.generate_transition_probabilities()
tagger.read_train_file()