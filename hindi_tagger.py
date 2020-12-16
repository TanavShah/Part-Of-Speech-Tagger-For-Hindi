from decimal import Decimal as dec
from tqdm import tqdm
import math
import numpy as np


class HindiTagger:
    train_file_name = 'train_set_hmm.csv'
    test_file_name = 'test_set_hmm.csv'

    # train_file_name = 'hmm_train_2.csv'
    # test_file_name = 'hmm_test_2.csv'

    p_word_tag = {}
    p_word = {}
    p_tag = {}
    dict_transition = {}
    cnt = 0

    def __init__(self):
        self.train()

    def generate_stem_words(self,word):
	    
        suffixes = {1: [u"ो",u"े",u"ू",u"ु",u"ी",u"ि",u"ा"], 2: [u"कर",u"ाओ",u"िए",u"ाई",u"ाए",u"ने",u"नी",u"ना",u"ते",u"ीं",u"ती",u"ता",u"ाँ",u"ां",u"ों",u"ें"], 3: [u"ाकर",u"ाइए",u"ाईं",u"ाया",u"ेगी",u"ेगा",u"ोगी",u"ोगे",u"ाने",u"ाना",u"ाते",u"ाती",u"ाता",u"तीं",u"ाओं",u"ाएं",u"ुओं",u"ुएं",u"ुआं"], 4: [u"ाएगी",u"ाएगा",u"ाओगी",u"ाओगे",u"एंगी",u"ेंगी",u"एंगे",u"ेंगे",u"ूंगी",u"ूंगा",u"ातीं",u"नाओं",u"नाएं",u"ताओं",u"ताएं",u"ियाँ",u"ियों",u"ियां"], 5: [u"ाएंगी",u"ाएंगे",u"ाऊंगी",u"ाऊंगा",u"ाइयाँ",u"ाइयों",u"ाइयां"]}
        
#         suffixes = {
# 1: [u"ो",u"े",u"ू",u"ु",u"ी",u"ि",u"ा",u"क"],
# 2: [u"कर",u"ाओ",u"िए",u"ाई",u"ाए",u"ने",u"नी",u"ना",u"ते",u"ीं",u"ती",u"ता",u"ाँ",u"ां",u"ों",u"ें",u"ाऊ",u"िक",u"ीय",u"ीच",u"ेद",u"ेय",u"कर",u"जी",u"तः",u"ता",u"त्व",u"पन"],
# 3: [u"ाकर",u"ाइए",u"ाईं",u"ाया",u"ेगी",u"ेगा",u"ोगी",u"ोगे",u"ाने",u"ाना",u"ाते",u"ाती",u"ाता",u"तीं",u"ाओं",u"ाएं",u"ुओं",u"ुएं",u"ुआं",u"ाना",u"ावा",u"िका",u"ियत",u"िया",u"ीला",u"कार",u"जनक",u"दान",u"दार",u"बाज़",u"वाद"],
# 4: [u"ाएगी",u"ाएगा",u"ाओगी",u"ाओगे",u"एंगी",u"ेंगी",u"एंगे",u"ेंगे",u"ूंगी",u"ूंगा",u"ातीं",u"नाओं",u"नाएं",u"ताओं",u"ताएं",u"ियाँ",u"ियों",u"ियां",u"ात्मक",u"ीकरण",u"कारक",u"गर्दी",u"गिरी",u"वादी",u"वाला",u"वाले",u"शाली",u"शुदा"],
# 5: [u"ाएंगी",u"ाएंगे",u"ाऊंगी",u"ाऊंगा",u"ाइयाँ",u"ाइयों",u"ाइयां"] }

#         suffixes = {
# 	    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],  
#             2: ["तृ","ान","ैत","ने","ाऊ","ाव","कर", "ाओ", "िए", "ाई", "ाए", "नी", "ना", "ते", "ीं", "ती",
#                 "ता", "ाँ", "ां", "ों", "ें","ीय", "ति","या", "पन", "पा","ित","ीन","लु","यत","वट","लू"],     
#             3: ["ेरा","त्व","नीय","ौनी","ौवल","ौती","ौता","ापा","वास","हास","काल","पान","न्त","ौना","सार","पोश","नाक",
#                 "ियल","ैया", "ौटी","ावा","ाहट","िया","हार", "ाकर", "ाइए", "ाईं", "ाया", "ेगी", "वान", "बीन",
#                 "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं","कला","िमा","कार",
#                 "गार", "दान","खोर"],     
#             4: ["ावास","कलाप","हारा","तव्य","वैया", "वाला", "ाएगी", "ाएगा", "ाओगी", "ाओगे", 
#                 "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां",
#                 "त्वा","तव्य","कल्प","िष्ठ","जादा","क्कड़"],     
#             5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां", "अक्कड़","तव्य:","निष्ठ"],
# }

# ret -> stem, suff, indicator
        ret = []

        for L in (5,4,3,2,1):
            if(len(word) >= L):
                for suff in suffixes[L]:
                    if (word.endswith(suff) and word != suff):
                        # ret.append(word[:-L])
                        # ret.append(suff)
                        # ret.append(1)

                        # return ret
                        return word[:-L]
        
        # ret.append(word)
        # ret.append(" ")
        # ret.append(0)

        # return ret
        return word                

    """
    Process dataset to break into sentences
    """

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
                    l = line.split("~")
                    s_list.append([l[0].strip(), l[1].strip()])
                    if train_data:
                        self.dict_transition[f"{l[1].strip()}_{prev}"] = self.dict_transition.get(
                            f"{l[1].strip()}_{prev}", 0) + 1
                    prev = l[1].strip()
        return vakya_list

    def train(self):

        vakya_list = self.process_input_file(self.train_file_name, train_data=True)
        cnt = 0
        for word_list in vakya_list:
            if(cnt == 0) :
                    print()
                    for word, tag in word_list :
                        print(word, end = " ")
                    print()
                    for word, tag in word_list :
                        print(self.generate_stem_words(word)[0], end = " ")
                    print()
                
            cnt += 1

            for word, tag in word_list:
                stem = self.generate_stem_words(word)
                # word = stem[0]
                # suffix = stem[1]
                # indicator = stem[2]
                word = self.generate_stem_words(word)
                self.p_word[word] = self.p_word.get(word, 0) + 1
                self.p_tag[tag] = self.p_tag.get(tag, 0) + 1
                self.p_word_tag[f"{word}_{tag}"] = self.p_word_tag.get(f"{word}_{tag}", 0) + 1

                # if(indicator == 1) :
                #     tag = "S" + tag
                #     self.p_word[suffix] = self.p_word.get(suffix, 0) + 1
                #     self.p_tag[tag] = self.p_tag.get(tag, 0) + 1
                #     self.p_word_tag[f"{suffix}_{tag}"] = self.p_word_tag.get(f"{suffix}_{tag}", 0) + 1

        for tag in self.p_tag:
            self.cnt += dec(self.dict_transition.get(f"{tag}_start", 0))

    def transition_prob(self, tag, prev_tag):
        if prev_tag == "start":
            return (dec(self.dict_transition.get(f"{tag}_{prev_tag}", 0)) + dec(1)) / (self.cnt + dec(len(self.dict_transition)))
        else:
            return (dec(self.dict_transition.get(f"{tag}_{prev_tag}", 0)) + dec(1)) / (dec(self.p_tag.get(prev_tag, 0)) + dec(len(self.dict_transition)))





    """
    Smmothing for new words
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

    
        """
        Rule Based Special Feature
        -> Check NN and NNP
        -> Check XC, JJ
        -> Check RDP
        """

        SYM = ['.', ',', '-', '"', '!', '/']

        cnt = 0

        for vakya in tqdm(self.process_input_file(self.test_file_name)):
            word_count += len(vakya)
            # words = []
            # index = np.zeros(2*len(vakya))
            # position = 0
            # for shabd in vakya :
            #     stem = self.generate_stem_words(shabd[0])
            #     words.append(stem[0])
            #     if (stem[2] == 1) :
            #         position += 1
            #         index[position] = 1
            #         words.append(stem[1])

            #     position += 1

            words = [self.generate_stem_words(shabd[0])[0] for shabd in vakya]

            if(cnt == 0) :
                print()
                for w in vakya :
                    print(w[0], end = " ")
                print()
                for w in words :
                    print(w, end = " ")
                    # print(w)
                print()

                print(len(vakya), len(words))
                

            predicted_tags = self.VITERBI(words)
            word_no = -1
            # for i in range(len(words)):
            for i in range(len(vakya)):
                # if(index[i]) :
                #     continue

                word_no += 1

                tag_predicted = predicted_tags[i]
                if(vakya[word_no][0].isnumeric()) :
                    tag_predicted = "QC"

                for j in SYM :
                    if(vakya[word_no][0] == j) :
                        tag_predicted = "SYM"
                        break

                if "-" in vakya[word_no][0] :
                    tag_predicted = "RDP"

                if(cnt == 0) :
                    print (tag_predicted, end = " ")

                tag = vakya[word_no][1]
                cm[inner_p_tag[tag]][inner_p_tag[tag_predicted]] += 1
            
            cnt += 1
            # break

        print()


        """
        Accuracy Measures
        """

        recall = []
        precision = []
        f_score = []

        pred_actual = 0
        for i in range(len(inner_p_tag)):
            pred_actual += cm[i][i]

        true_positive = 0
        false_negative = 0
        false_positive = 0
        total_true_positive = 0
        total_false_positive = 0
        total_false_negative = 0

        total_recall = 0
        total_precision = 0

        for i in range(len(inner_p_tag)):
            true_positive = cm[i][i]
            false_positive = 0
            false_negative = 0

            for j in range(len(inner_p_tag)):
                if(i == j) :
                    continue

                false_negative += cm[i][j]
                false_positive += cm[j][i]

            total_true_positive += true_positive
            total_false_positive += false_positive
            total_false_negative += false_negative

            if(true_positive == 0) :
                recall.append(0)
                precision.append(0)
                f_score.append(0)
            else :
                recall.append(true_positive/(true_positive + false_negative))
                precision.append(true_positive/(true_positive + false_positive))
                f_score.append((2*recall[i]*precision[i])/(recall[i] + precision[i]))
            
            total_recall += recall[i]
            total_precision += precision[i]

        micro_precision = total_true_positive/(total_true_positive + total_false_positive)
        micro_recall = total_true_positive/(total_true_positive + total_false_negative)
        micro_f_score = ((2*micro_precision*micro_recall)/(micro_precision + micro_recall))

        macro_precision = total_precision/22
        macro_recall = total_recall/22
        macro_f_score = ((2*macro_precision*macro_recall)/(macro_precision + macro_recall))

        print("Test Accuracy : ", pred_actual/word_count)
        print()
        print("Confusion Matrix : ")
        for i in range(len(cm)) :
            for j in range(len(cm[0])) :
                leng = 0
                if(cm[i][j] != 0) :
                    leng = int(math.log10(cm[i][j]))
                print((4 - leng)*" ", end = "")
                print(cm[i][j], end = " ")
            print()

        print()

        print ("Recall for each tag : ")
        cnt = 0
        for i in (inner_p_tag):
            print(i, " : ", recall[cnt])
            cnt += 1
	
        print()

        print ("Precision for each tag : ")
        cnt = 0
        for i in (inner_p_tag):
            print(i, " : ", precision[cnt])
            cnt += 1

        print()

        print ("F-measure for each tag : ")
        cnt = 0
        for i in (inner_p_tag):
            print(i, " : ", f_score[cnt])
            cnt += 1

        print()

        print ("Micro Measures : ")
        print ("Recall : ", micro_recall)
        print ("Precision : ", micro_precision)
        print ("F-measure : ", micro_f_score)

        print()

        print ("Macro Measures : ")
        print ("Recall : ", macro_recall)
        print ("Precision : ", macro_precision)
        print ("F-measure : ", macro_f_score)

        # print(self.generate_stem_words("लाल-वाल"))


    def predict(self):
        self.hmm_bi_gram()


def main():
    tagger = HindiTagger()
    # tagger.p_tag["NN"] = 94664
    # tagger.p_tag["PSP"] = 66765
    # tagger.p_tag["SYM"] = 38376
    # tagger.p_tag["JJ"] = 35687
    # tagger.p_tag["VM"] = 31352
    # tagger.p_tag["XC"] = 26955
    # tagger.p_tag["NNP"] = 26780
    # tagger.p_tag["QC"] = 20651
    # tagger.p_tag["VAUX"] = 14934
    # tagger.p_tag["PRP"] = 12614
    # tagger.p_tag["CC"] = 9258
    # tagger.p_tag["RP"] = 3925
    # tagger.p_tag["DEM"] = 3435
    # tagger.p_tag["NST"] = 3151
    # tagger.p_tag["QF"] = 2146
    # tagger.p_tag["RB"] = 1042
    # tagger.p_tag["QO"] = 937
    # tagger.p_tag["INTF"] = 895
    # tagger.p_tag["NEG"] = 580
    # tagger.p_tag["RDP"] = 439
    # tagger.p_tag["WQ"] = 81
    # tagger.p_tag["INJ"] = 5










    tagger.predict()

    input_sentence = "सलेमपुर तेरा बांगर कन्नौज , कन्नौज , उत्तर प्रदेश स्थित एक गांव है ."
    vakya = []

    vakya = input_sentence.split(" ")
    
    print (vakya)

    print(tagger.VITERBI(vakya))

if __name__ == "__main__":
    main()
