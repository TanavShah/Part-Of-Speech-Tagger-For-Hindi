import glob
import xml.etree.ElementTree as ET
from decimal import Decimal
from typing import Iterator
import pickle
import os
import sys
import csv


WORD_TAG_DICT =dict()
WORD_DICT = dict()
TAG_DICT = dict()
TAG_TRANSITION = dict()
start_count = Decimal(0)


def parse_sentence(sentence: list, train: bool) -> list:
    global TAG_TRANSITION
    prev_tags = ["^"]
    word_list = list()
    skip_cnt = 0
    for word in sentence.findall('.//*[@c5]'):
        if skip_cnt > 0:
            skip_cnt -= 1
            continue
        if word.tag != "mw":
            try:
                text = word.text.strip()
                tags = word.get("c5").split("-")
            except:
                continue
        else:
            text = ""
            for w in word:
                skip_cnt += 1
                text += w.text
            text = text.strip()
            tags = word.get("c5").split("-")


        word_list.append([text, tags])
        if train:
            for prev_tag in prev_tags:
                for tag in tags:
                    TAG_TRANSITION[f"{tag}_{prev_tag}"] = TAG_TRANSITION.get(f"{tag}_{prev_tag}", 0) + 1
        prev_tags = tags
    return word_list


def parse_single_xml(xml_fname: str, train=False) -> Iterator:
    with open(xml_fname) as handle:
        sentence = []
        prev = "^"
        for idx, line in enumerate(handle):
            if idx == 0:
                continue
            if line.strip() == "~":
                yield sentence
                sentence = []
                prev = "^"
            else:
                l = line.split('~')
                # print(l)
                sentence.append([l[0].strip(), [l[1].strip()]])
                if train:
                    TAG_TRANSITION[f"{l[1].strip()}_{prev}"] = TAG_TRANSITION.get(f"{l[1].strip()}_{prev}", 0) + 1
                prev = l[1].strip()
    # tree = ET.parse(xml_fname)
    # sentences = tree.findall(".//s") # get all sentences

    # # sentence_list = list()
    # for sentence in sentences:
    #     tmp = parse_sentence(sentence, train)
    #     if not tmp:
    #         print(xml_fname)
    #         print(sentence.get('n'))
    #         exit(1)
    #     yield tmp
    # return sentence_list


def train(train_files_list : list):
    global WORD_TAG_DICT
    global WORD_DICT
    global TAG_DICT
    global TAG_TRANSITION
    global start_count

    if os.path.exists("./cache"):
        with open('./cache', 'rb') as f:
            WORD_TAG_DICT, WORD_DICT, TAG_DICT, TAG_TRANSITION, start_count = pickle.load(f)
        return

    for fname in train_files_list:
        sentence_list = parse_single_xml(fname, train=True)
        for word_list in sentence_list:
            for word, tags in word_list:
                for tag in tags:
                    # print(word, tag)
                    # exit(1)
                    WORD_DICT[word] = WORD_DICT.get(word, 0) + 1
                    TAG_DICT[tag] = TAG_DICT.get(tag, 0) + 1
                    WORD_TAG_DICT[f"{word}_{tag}"] = WORD_TAG_DICT.get(f"{word}_{tag}", 0) + 1

    for tag in TAG_DICT:
        start_count += Decimal(TAG_TRANSITION.get(f"{tag}_^", 0))
    with open('./cache', 'wb') as f:
        pickle.dump([WORD_TAG_DICT, WORD_DICT, TAG_DICT, TAG_TRANSITION, start_count], f)


def probability_tag_tag(tag: str, prev_tag: str) -> Decimal:
    # Add one smoothing
    if prev_tag == "^":
        # start probabilities
        return (Decimal(TAG_TRANSITION.get(f"{tag}_{prev_tag}", 0)) + Decimal(1)) / (start_count + Decimal(len(TAG_TRANSITION)))
    else:
        # transition probabilities
        return (Decimal(TAG_TRANSITION.get(f"{tag}_{prev_tag}", 0)) + Decimal(1)) / (Decimal(TAG_DICT.get(prev_tag, 0)) + Decimal(len(TAG_TRANSITION)))


def probability_word_tag(word: str, tag: str) -> Decimal:
    # add 1 smoothing
    return ( Decimal(WORD_TAG_DICT.get(f"{word}_{tag}", 0)) + Decimal(1) ) / ( Decimal(TAG_DICT.get(tag, 0)) + Decimal(len(TAG_DICT)) )


def viterbi(sentence: list) -> list:
    assert len(sentence) > 0

    # probability_matrix = [{key: Decimal(0.0) for key in TAG_DICT} for _ in range(len(sentence))]
    # back_ptr = [{key: None for key in TAG_DICT} for _ in range(len(sentence))]

    probability_matrix = list()
    back_ptr = list()

    for _ in range(len(sentence)):
        matrix_temp = dict()
        back_tmp = dict()
        for key in TAG_DICT:
            matrix_temp[key] = Decimal(0)
            back_tmp[key] = None
        probability_matrix.append(matrix_temp)
        back_ptr.append(back_tmp)

    for tag in TAG_DICT:
        probability_matrix[0][tag] = probability_tag_tag(tag, "^") * probability_word_tag(sentence[0], tag)
        back_ptr[0][tag] = None

    for idx in range(len(sentence)):
        for tag in TAG_DICT:
            for prev_tag in TAG_DICT:
                back_probability = probability_matrix[idx - 1][prev_tag] * probability_tag_tag(tag, prev_tag) * probability_word_tag(sentence[idx], tag)
                if  back_probability > probability_matrix[idx][tag]:
                    probability_matrix[idx][tag] = back_probability
                    back_ptr[idx][tag] = prev_tag

    best_probability = max(probability_matrix[idx], key=probability_matrix[idx].get)
    sequence = list()
    iterator = best_probability
    while iterator is not None:
        sequence.append(iterator)
        iterator = back_ptr[idx][iterator]
        idx -= 1
    sequence.reverse()
    assert len(sequence) == len(sentence)
    return sequence


def hmm(test_files_list: list, f_start: int, f_end: int):
    tag_dict = dict()
    idx = 0
    for key in TAG_DICT.keys():
        tag_dict[key] = idx
        idx += 1


    confusion_matrix = list()

    for i in range(len(tag_dict.keys()) + 1):
        t = list()
        for j in range(idx + 1):
            t.append(0)
        confusion_matrix.append(t)

    word_count = 0
    f_cnt = 0
    for fname in test_files_list:
        # skip initial `f_skip` number of files
        # FOR TESTING PURPOSE ONLY
        # TODO : remove this at the end
        if f_cnt < f_start:
            f_cnt += 1
            continue
        print(fname)
        for sentence in parse_single_xml(fname):
            word_count += len(sentence)
            words = [word[0] for word in sentence]
            predicted_tags = viterbi(words)
            for i in range(len(sentence)):
                tag_predicted = predicted_tags[i]
                for tag in sentence[i][1]:
                    confusion_matrix[tag_dict[tag]][tag_dict[tag_predicted]] = confusion_matrix[tag_dict[tag]][tag_dict[tag_predicted]] + 1
            # break
         # break
        f_cnt += 1
        if f_cnt >= f_end:
            break


    # for i in range(len(tag_dict.keys())):
    #     sum_row = 0
    #     for j in range(len(tag_dict.keys())):
    #         sum_row += confusion_matrix[i][j]
    
    #     confusion_matrix[i][idx] = sum_row
    
    # for i in range(len(tag_dict.keys())):
    #     sum_col = 0
    #     for j in range(len(tag_dict.keys())):
    #         sum_col += confusion_matrix[j][i]
    
    correct_pred = 0
    for i in range(len(tag_dict.keys())):
        correct_pred += confusion_matrix[i][i]

    print (correct_pred, word_count)
    print ("The Accuracy on Test-Set : ", end = " ")
    print (correct_pred/(correct_pred + word_count))

def main():
    train(['train_set_hmm.csv'])
    hmm(['test_set_hmm.csv'], 0, 0)


if __name__ == "__main__":
    main()

