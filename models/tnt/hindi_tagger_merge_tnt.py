from decimal import Decimal as dec
from pathlib import Path
from tqdm import tqdm
from nltk.tag import tnt 
import math

class HindiTagger:
    train_file_name = Path(__file__).parent / '../../dataset/stemming/train_set.csv'
    test_file_name = Path(__file__).parent /'../../dataset/stemming/dev_set.csv'

    p_word_tag = {}
    p_word = {}
    p_tag = {}
    tag_list = ['DEM', 'NNP', 'PSP', 'INTF', 'JJ', 'NN', 'QC', 'VM', 'SYM', 'PRP', 'NNC', 'VAUX', 'CC', 'NNPC', 'NST', 'RP', 'NSTC', 'QF', 'JJC', 'WQ', 'NEG', 'RB', 'RDP', 'RBC', 'QO', 'CCC', 'QCC', 'UNK', 'PRPC', 'QFC', 'INJ']
    dict_transition = {}
    cnt = 0

    def generate_stem_words(self, word):
        suffixes = {
            1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
            2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
            3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
            4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
            5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"]
        }

        for L in (5,4,3,2,1):
            if(len(word) >= L):
                for suf in suffixes[L]:
                    if(word.endswith(suf) and word!=suf):
                        return word[:-L]

        return word

    def data_tuples(self, file_name, stem):
        vakya_list = []
        with open(file_name) as inputFile:
            s_list = []
            for idx, line in enumerate(inputFile):
                if idx == 0:
                    continue
                if line.strip() == "~":
                    vakya_list.append(s_list)
                    s_list = []
                else:
                    l = line.split('~')
                    if(stem is False):
                        s_list.append((l[0].strip(), l[1].strip()))
                        if(self.p_word.get(l[0].strip()) is None):
                            self.p_word[l[0].strip()] = 0
                        self.p_word[l[0].strip()] += 1    
                    else:                
                        s_list.append((self.generate_stem_words(l[0].strip()), l[1].strip()))            
        return vakya_list


    def process_input_file(self, file_name, train_data=False):
        vakya_list = []
        with open(file_name) as inputFile:
            s_list = []
            prev = "start"
            for idx, line in enumerate(inputFile):
                if idx == 0:
                    continue
                if line.strip() == "~":
                    vakya_list.append(s_list)
                    s_list = []
                    prev = "start"
                else:
                    l = line.split('~')
                    s_list.append([l[0].strip(), l[1].strip()])
                    if train_data:
                        self.dict_transition[f"{l[1].strip()}_{prev}"] = self.dict_transition.get(
                            f"{l[1].strip()}_{prev}", 0) + 1
                    prev = l[1].strip()
        return vakya_list



def main():
    tagger = HindiTagger()
    # tagger.predict()

    train_data = tagger.data_tuples(tagger.train_file_name, stem=False)
    train_data_stem = tagger.data_tuples(tagger.train_file_name, stem=True)

    test_data = tagger.data_tuples(tagger.train_file_name, stem=False)
    test_data_stem = tagger.data_tuples(tagger.train_file_name, stem=True)

    tnt_tagging = tnt.TnT()
    tnt_tagging.train(train_data)

    tnt_tagging_stem = tnt.TnT()
    tnt_tagging_stem.train(train_data_stem)

    inner_p_tag = {}
    index = 0
    for key in tagger.tag_list:
        inner_p_tag[key] = index
        index += 1

    cm = []

    for tag_dict_index in range(len(inner_p_tag.keys())):
        abc = []
        for j in range(index):
            abc.append(0)
        cm.append(abc)

    #####

    inner_p_tag_1 = {}
    index_1 = 0
    for key in tagger.tag_list:
        inner_p_tag_1[key] = index_1
        index_1 += 1

    cm_1 = []

    for tag_dict_index in range(len(inner_p_tag_1.keys())):
        abc = []
        for j in range(index_1):
            abc.append(0)
        cm_1.append(abc)    

    word_count = 0

    ######

    inner_p_tag_2 = {}
    index_2 = 0
    for key in tagger.tag_list:
        inner_p_tag_2[key] = index_2
        index_2 += 1

    cm_2 = []

    for tag_dict_index in range(len(inner_p_tag_2.keys())):
        abc = []
        for j in range(index_2):
            abc.append(0)
        cm_2.append(abc)    

    #####

    word_count = 0

    for vakya in tqdm(tagger.process_input_file(tagger.test_file_name)):
        word_count += len(vakya)
        
        words = [shabd[0] for shabd in vakya]
        words_stem = [tagger.generate_stem_words(shabd[0]) for shabd in vakya]

        predicted_tags_1 = [tup[1].upper() for tup in tnt_tagging.tag(words)]
        predicted_tags_2 = [tup[1].upper() for tup in tnt_tagging_stem.tag(words_stem)]

        merge_tags = list()

        for index, word in enumerate(words):
            final_tag = None
            
            if(tagger.p_word.get(word)):
                final_tag = predicted_tags_1[index]
            else:
                final_tag = predicted_tags_2[index]    

            merge_tags.append(final_tag)    


        for i in range(len(vakya)):
            tag_predicted = merge_tags[i]
            tag = vakya[i][1]
            cm[inner_p_tag[tag]][inner_p_tag[tag_predicted]] += 1

        for i in range(len(vakya)):
            tag_predicted = predicted_tags_1[i]
            tag = vakya[i][1]
            cm_1[inner_p_tag_1[tag]][inner_p_tag_1[tag_predicted]] += 1

        for i in range(len(vakya)):
            tag_predicted = predicted_tags_2[i]
            tag = vakya[i][1]
            cm_2[inner_p_tag_2[tag]][inner_p_tag_2[tag_predicted]] += 1                    

    pred_actual = 0
    for i in range(len(inner_p_tag)):
        pred_actual += cm[i][i]

    print(pred_actual, word_count)
    print("Accuracy Merge: ", end=" ")
    print(pred_actual / word_count)
    print(cm)

    print("-------------------------------")

    pred_actual_1 = 0
    for i in range(len(inner_p_tag_1)):
        pred_actual_1 += cm_1[i][i]

    print(pred_actual_1, word_count)
    print("Accuracy without stem: ", end=" ")
    print(pred_actual_1 / word_count)
    print(cm_1)

    print("-------------------------------")

    pred_actual_2 = 0
    for i in range(len(inner_p_tag_2)):
        pred_actual_2 += cm_2[i][i]

    print(pred_actual_2, word_count)
    print("Accuracy with stem: ", end=" ")
    print(pred_actual_2 / word_count)
    print(cm_2)
        

if __name__ == "__main__":
    main()
