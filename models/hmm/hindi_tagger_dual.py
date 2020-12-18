from decimal import Decimal as dec
from pathlib import Path
from tqdm import tqdm

class HindiTagger:
    train_file_name = Path(__file__).parent / '../../dataset/stemming/train_set.csv'
    test_file_name = Path(__file__).parent /'../../dataset/stemming/dev_set.csv'

    p_word_tag = {}
    p_word = {}
    p_tag = {}
    dict_transition = {}
    cnt = 0

    stem = False

    def __init__(self, stem):
        self.stem = stem
        self.train()

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
                    if self.stem is True:
                        print(self.generate_stem_words(word), end = " ")
                    else:
                        print(word, end = " ")    
                print()

            count+=1


            for word, tag in word_list:
                if self.stem is True:
                    word = self.generate_stem_words(word)
                self.p_word[word] = self.p_word.get(word, 0) + 1
                self.p_tag[tag] = self.p_tag.get(tag, 0) + 1
                self.p_word_tag[f"{word}_{tag}"] = self.p_word_tag.get(f"{word}_{tag}", 0) + 1

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
            viterbi_table[0][t] = self.transition_prob(t, "start") * self.emission_prob(vakya[0],
                                                                                        t)
            prev_column[0][t] = None

        for idx in range(len(vakya)):
            for t in self.p_tag:
                for t_dash in self.p_tag:
                    prev_prob = viterbi_table[idx - 1][t_dash] * self.transition_prob(t, t_dash) * self.emission_prob(
                        vakya[idx], t)
                    if prev_prob > viterbi_table[idx][t]:
                        viterbi_table[idx][t] = prev_prob
                        prev_column[idx][t] = t_dash

        max_prob = max(viterbi_table[idx], key=viterbi_table[idx].get)

        seq = []
        itr = max_prob
        while itr is not None:
            seq.append(itr)
            itr = prev_column[idx][itr]
            idx -= 1
        seq = seq[::-1]
        return seq

    def hmm_bi_gram(self, vakya):
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
        
        word_count += len(vakya)
        if(self.stem is True):
            words = [self.generate_stem_words(shabd[0]) for shabd in vakya]
        else:                
            words = [shabd[0] for shabd in vakya]

        predicted_tags = self.VITERBI(words)
        return predicted_tags

    def predict(self, vakya):
        return self.hmm_bi_gram(vakya)


def main():
    input_sentence = "सलेमपुर तेरा बांगर कन्नौज , कन्नौज , उत्तर प्रदेश स्थित एक गांव है ."
    
    tagger = HindiTagger(stem=False)
    tagger_stem = HindiTagger(stem=True)

    print("Prediction without stem:")
    print(tagger.predict(input_sentence))

    print("Prediction with stem:")
    print(tagger_stem.predict(input_sentence))

    inner_p_tag = {}
    index = 0
    for key in tagger.p_tag.keys():
        inner_p_tag[key] = index
        index += 1

    cm = []

    for tag_dict_index in range(len(inner_p_tag.keys())):
        tnt = []
        for j in range(index):
            tnt.append(0)
        cm.append(tnt)

    #####

    inner_p_tag_1 = {}
    index_1 = 0
    for key in tagger.p_tag.keys():
        inner_p_tag_1[key] = index_1
        index_1 += 1

    cm_1 = []

    for tag_dict_index in range(len(inner_p_tag_1.keys())):
        tnt = []
        for j in range(index_1):
            tnt.append(0)
        cm_1.append(tnt)    

    word_count = 0

    ######

    inner_p_tag_2 = {}
    index_2 = 0
    for key in tagger.p_tag.keys():
        inner_p_tag_2[key] = index_2
        index_2 += 1

    cm_2 = []

    for tag_dict_index in range(len(inner_p_tag_2.keys())):
        tnt = []
        for j in range(index_2):
            tnt.append(0)
        cm_2.append(tnt)    

    #####

    word_count = 0

    for vakya in tqdm(tagger.process_input_file(tagger.test_file_name)):
        word_count += len(vakya)
        
        words = [shabd[0] for shabd in vakya]

        predicted_tags_1 = tagger.predict(vakya)
        predicted_tags_2 = tagger_stem.predict(vakya)

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
