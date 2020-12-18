from decimal import Decimal as dec
from pathlib import Path
from tqdm import tqdm

class HindiTagger:
    train_file_name = Path(__file__).parent / '../../dataset/hmm/hmm_train_2.csv'
    test_file_name = Path(__file__).parent /'../../dataset/hmm/hmm_test_2.csv'

    p_word_tag = {}
    p_word = {}
    p_tag = {}
    dict_transition = {}
    cnt = 0

    def __init__(self):
        self.train()

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
        for word_list in vakya_list:
            for word, tag in word_list:
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

        for vakya in tqdm(self.process_input_file(self.test_file_name)):
            word_count += len(vakya)
            words = [shabd[0] for shabd in vakya]
            predicted_tags = self.VITERBI(words)
            for i in range(len(vakya)):
                tag_predicted = predicted_tags[i]
                tag = vakya[i][1]
                cm[inner_p_tag[tag]][inner_p_tag[tag_predicted]] += 1

        pred_actual = 0
        for i in range(len(inner_p_tag)):
            pred_actual += cm[i][i]

        print(pred_actual, word_count)
        print(cm)

    def predict(self):
        self.hmm_bi_gram()


def main():
    tagger = HindiTagger()
    tagger.predict()


if __name__ == "__main__":
    main()
