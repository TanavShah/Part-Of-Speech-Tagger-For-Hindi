from decimal import Decimal as dec
from pathlib import Path
from tqdm import tqdm
import csv
import math

class HindiTagger:
    train_file_name = Path(__file__).parent / '../../dataset/stemming/train_set.csv'
    test_file_name = Path(__file__).parent /'../../dataset/stemming/test_set.csv'

    p_word_tag = {}
    p_stem_tag = {}
    p_word = {}
    p_tag = {}
    dict_transition = {}
    cnt = 0
    d1 = dict() 
    d2 = dict()
    def __init__(self):
        self.train()

    def generate_stem_words(self, word):
        
        suffixes = {
            1: [u"ो",u"े",u"ू",u"ु",u"ी",u"ि",u"ा",u"क"],
            2: [u"कर",u"ाओ",u"िए",u"ाई",u"ाए",u"ने",u"नी",u"ना",u"ते",u"ीं",u"ती",u"ता",u"ाँ",u"ां",u"ों",u"ें",u"ाऊ",u"िक",u"ीय",u"ीच",u"ेद",u"ेय",u"कर",u"जी",u"तः",u"ता",u"त्व",u"पन"],
            3: [u"ाकर",u"ाइए",u"ाईं",u"ाया",u"ेगी",u"ेगा",u"ोगी",u"ोगे",u"ाने",u"ाना",u"ाते",u"ाती",u"ाता",u"तीं",u"ाओं",u"ाएं",u"ुओं",u"ुएं",u"ुआं",u"ाना",u"ावा",u"िका",u"ियत",u"िया",u"ीला",u"कार",u"जनक",u"दान",u"दार",u"बाज़",u"वाद"],
            4: [u"ाएगी",u"ाएगा",u"ाओगी",u"ाओगे",u"एंगी",u"ेंगी",u"एंगे",u"ेंगे",u"ूंगी",u"ूंगा",u"ातीं",u"नाओं",u"नाएं",u"ताओं",u"ताएं",u"ियाँ",u"ियों",u"ियां",u"ात्मक",u"ीकरण",u"कारक",u"गर्दी",u"गिरी",u"वादी",u"वाला",u"वाले",u"शाली",u"शुदा"],
            5: [u"ाएंगी",u"ाएंगे",u"ाऊंगी",u"ाऊंगा",u"ाइयाँ",u"ाइयों",u"ाइयां"] }

        for L in (5,4,3,2,1):
            if(len(word) >= L):
                for suf in suffixes[L]:
                    if(word.endswith(suf) and word!=suf):
                        return word[:-L]

        return word

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

    def train(self):

        vakya_list = self.process_input_file(self.train_file_name, train_data=True)
        count = 0
        for word_list in vakya_list:
            if(count==0):
                print()
                for word, tag in word_list:
                    print(word, end = " ")
                print()
                for word, tag in word_list:
                    print(self.generate_stem_words(word), end = " ")
                print()

            count+=1


            for word, tag in word_list:
                self.p_word[word] = self.p_word.get(word, 0) + 1
                self.p_tag[tag] = self.p_tag.get(tag, 0) + 1
                self.p_word_tag[f"{word}_{tag}"] = self.p_word_tag.get(f"{word}_{tag}", 0) + 1

                stem = self.generate_stem_words(word)
                self.p_stem_tag[f"{stem}_{tag}"] = self.p_stem_tag.get(f"{stem}_{tag}", 0) + 1

        for tag in self.p_tag:
            self.cnt += dec(self.dict_transition.get(f"{tag}_start", 0))

    def transition_prob(self, tag, prev_tag):
        if prev_tag == "start":
            return (dec(self.dict_transition.get(f"{tag}_{prev_tag}", 0)) + dec(1)) / (self.cnt + dec(len(self.dict_transition)))
        else:
            return (dec(self.dict_transition.get(f"{tag}_{prev_tag}", 0)) + dec(1)) / (dec(self.p_tag.get(prev_tag, 0)) + dec(len(self.dict_transition)))


    """
    Smoothing for new words
    """

    def emission_prob(self, word, tag):
        return (dec(self.p_word_tag.get(f"{word}_{tag}", 0)) + dec(1)) / (
                dec(self.p_tag.get(tag, 0)) + dec(len(self.p_tag)))

    def emission_prob_stemming(self, word, tag):
        return (dec(self.p_stem_tag.get(f"{word}_{tag}", 0)) + dec(1)) / (
                dec(self.p_tag.get(tag, 0)) + dec(len(self.p_tag)))

    def VITERBI(self, vakya):
        viterbi_table = []
        prev_column = []

        for value in range(len(vakya)):
            temp_table = {}
            temp_prev = {}
            for key in self.p_tag:
                temp_table[key] = 0
                temp_prev[key] = None
            viterbi_table.append(temp_table)
            prev_column.append(temp_prev)

        for t in self.p_tag:
            
            viterbi_table[0][t] = 0
            if(self.p_word.get(vakya[0])):
                viterbi_table[0][t] = self.transition_prob(t, "start") * self.emission_prob(vakya[0], t)
            else :
                viterbi_table[0][t] = self.transition_prob(t, "start") * self.emission_prob_stemming(self.generate_stem_words(vakya[0]), t)

            prev_column[0][t] = None

        # cnt = 0
        # cnt1 = 0
        for idx in range(len(vakya)):
            # if(self.p_word.get(vakya[idx])):
            #     cnt1 += 1
            # else :
            #     cnt += 1
            for t in self.p_tag:
                for t_dash in self.p_tag:
                    prev_prob = 0
                    if(self.p_word.get(vakya[idx])):
                        prev_prob = viterbi_table[idx - 1][t_dash] * self.transition_prob(t, t_dash) * self.emission_prob(vakya[idx], t)
                    else :
                        prev_prob = viterbi_table[idx - 1][t_dash] * self.transition_prob(t, t_dash) * self.emission_prob_stemming(self.generate_stem_words(vakya[idx]), t)

                    if prev_prob > viterbi_table[idx][t]:
                        viterbi_table[idx][t] = prev_prob
                        prev_column[idx][t] = t_dash

        max_prob = max(viterbi_table[idx], key=viterbi_table[idx].get)
        # print(f"cnt {cnt}")
        # print (f"cnt1 {cnt1}")
        seq = []
        itr = max_prob
        while itr is not None:
            seq.append(itr)
            itr = prev_column[idx][itr]
            idx -= 1
        seq = seq[::-1]
        return seq

    def hmm_bi_gram(self):
        inner_p_tag = {}
        index = 0
        for key in self.p_tag.keys():
            inner_p_tag[key] = index
            index += 1

        cm = []

        for tag_dict_index in range(len(inner_p_tag.keys())):
            tnt = []
            for j in range(index):
                tnt.append(0)
            cm.append(tnt)

        word_count = 0
        sentence_count = 0
        for vakya in tqdm(self.process_input_file(self.test_file_name)):
            word_count += len(vakya)
            words = [shabd[0] for shabd in vakya]

            # if(sentence_count < 5):
            #     print(sentence_count)
            #     print(vakya)
            #     print(words)
            #     print("--------------")

            predicted_tags = self.VITERBI(words)
            for i in range(len(vakya)):
                tag_predicted = predicted_tags[i]
                tag = vakya[i][1]
                cm[inner_p_tag[tag]][inner_p_tag[tag_predicted]] += 1

            sentence_count += 1    
            # break

        pred_actual = 0
        for i in range(len(inner_p_tag)):
            pred_actual += cm[i][i]

        tags_list = []
        for i in inner_p_tag :
            tags_list.append(i)

        print(pred_actual, word_count)
        print("Test Accuracy : ", pred_actual/word_count)
        print()
        print("Confusion Matrix : ")
        for i in range(len(cm)) :
            print(tags_list[i], end = "")
            print((5 - len(tags_list[i]))*" ", end = "")
            for j in range(len(cm[0])) :
                leng = 0
                if(cm[i][j] != 0) :
                    leng = int(math.log10(cm[i][j]))
                print((4 - leng)*" ", end = "")
                print(cm[i][j], end = " ")
            print()

        print()

    def predict(self):
        self.hmm_bi_gram()


def main():
    tagger = HindiTagger()
    tagger.predict()


if __name__ == "__main__":
    main()
