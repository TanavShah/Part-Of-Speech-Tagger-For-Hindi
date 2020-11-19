import csv
import json
import os.path
from os import path
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# tag_dict: <word: max_tax_freq> {"word": {"tag1": freq ... }}
# generate dict
# get_word

class Tagger:

    __train_file_name = "train_set.csv"
    __test_file_name = "test_set.csv"
    __tag_dict_file_name = "model.json"
    __tags_frequency_file_name = "tags_frequency.csv"

    tag_dict = None

    tags_list = None
    tags_prob_list = None

    y_predicted = []
    y_actual = []

    def __init__(self):
        self.generate_dict()

    def check_dict(self):
        if (path.exists(self.__tag_dict_file_name) == True):
            with open(self.__tag_dict_file_name, "r") as jsonFile:
                content = jsonFile.read()
                self.tag_dict = json.loads(content)
        else:
            self.generate_dict()

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
    
    cnt = 0
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
            self.cnt = self.cnt + 1
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
                prediction = self.get_best_tag(entry[0], mode)
                actual = entry[1]
                preds.append(prediction)
                actuals.append(actual)
            self.y_predicted = preds
            self.y_actual = actuals    
        print("Test accuracy: %.5f" % self.find_accuracy())    

    def confusion_matrix(self) :
        return confusion_matrix(self.y_actual, self.y_predicted, labels=self.tags_list)
    
    def find_accuracy(self) :
        return accuracy_score(self.y_actual, self.y_predicted)

print("Running")
tagger = Tagger()
tagger.evaluate("max")
tagger.evaluate("weighted")

# print(tagger.confusion_matrix()[0])
print(np.shape(tagger.confusion_matrix()))
print(tagger.cnt)