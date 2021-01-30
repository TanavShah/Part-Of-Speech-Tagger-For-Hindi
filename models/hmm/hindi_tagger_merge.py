from decimal import Decimal as dec
from pathlib import Path
from tqdm import tqdm
from hindi_tagger_decisive_viterbi import HindiTaggerDecisiveViterbi
from hindi_tagger_no_stem import HindiTaggerNoStem
from hindi_tagger_full_stem import HindiTaggerFullStem


def main():
    
    tagger_decisive = HindiTaggerDecisiveViterbi()
    tagger_no_stem = HindiTaggerNoStem()
    tagger_stem = HindiTaggerFullStem()

    cm, cm_no_stem, cm_stem, word_count = merge(tagger_no_stem, tagger_stem)

    print_results(cm, word_count, "Merge")
    print_results(cm_no_stem, word_count, "Without Stemming")
    print_results(cm_stem, word_count, "With Stemming")

    cm, word_count = evaluate(tagger_decisive)
    
    print_results(cm, word_count, "Decisive Viterbi")

def merge(tagger_no_stem, tagger_stem):
    tag_index_map = {}

    for idx, key in enumerate(tagger_no_stem.p_tag.keys()):
        tag_index_map[key] = idx

    total_tags = len(tag_index_map)

    cm = [[0 for i in range(total_tags)] for j in range(total_tags)]
    cm_no_stem = [[0 for i in range(total_tags)] for j in range(total_tags)]
    cm_stem = [[0 for i in range(total_tags)] for j in range(total_tags)]


    word_count = 0

    for vakya in tqdm(tagger_no_stem.process_input_file(tagger_no_stem.test_file_name)):

        vakya_len = len(vakya)
        word_count += vakya_len

        words = [tup[0] for tup in vakya]

        predicted_tags_no_stem = tagger_no_stem.predict(vakya)
        predicted_tags_stem = tagger_stem.predict(vakya)

        merge_tags = list()

        for index, word in enumerate(words):
            final_tag = None
            
            if(tagger_no_stem.p_word.get(word)):
                final_tag = predicted_tags_no_stem[index]
            else:
                final_tag = predicted_tags_stem[index]    

            merge_tags.append(final_tag)    


        for i in range(vakya_len):
            tag_actual = vakya[i][1]
            tag_actual_index = tag_index_map[tag_actual]

            tag_pred_merge_index = tag_index_map[merge_tags[i]]
            tag_pred_1_index = tag_index_map[predicted_tags_no_stem[i]]
            tag_pred_2_index = tag_index_map[predicted_tags_stem[i]]

            cm[tag_actual_index][tag_pred_merge_index] += 1
            cm_no_stem[tag_actual_index][tag_pred_1_index] += 1
            cm_stem[tag_actual_index][tag_pred_2_index] += 1


    return (cm, cm_no_stem, cm_stem, word_count)        

# Common Functions

def evaluate(tagger):
    tag_index_map = {}

    for idx, key in enumerate(tagger.p_tag.keys()):
        tag_index_map[key] = idx

    total_tags = len(tag_index_map)

    cm = [[0 for i in range(total_tags)] for j in range(total_tags)]
    cm_no_stem = [[0 for i in range(total_tags)] for j in range(total_tags)]
    cm_stem = [[0 for i in range(total_tags)] for j in range(total_tags)]


    word_count = 0

    for vakya in tqdm(tagger.process_input_file(tagger.test_file_name)):

        vakya_len = len(vakya)
        word_count += vakya_len

        words = [tup[0] for tup in vakya]

        predicted_tags = tagger.predict(vakya)
    
        for i in range(vakya_len):
            tag_actual = vakya[i][1]
            tag_actual_index = tag_index_map[tag_actual]

            tag_pred_index = tag_index_map[predicted_tags[i]]
            
            cm[tag_actual_index][tag_pred_index] += 1

    return (cm, word_count)        


def print_results(confusion_matrix: list, word_count: int, type: str):
    pred_actual = 0
    for i in range(len(confusion_matrix[0])):
        pred_actual += confusion_matrix[i][i]

    print(pred_actual, word_count)

    print(f"Accuracy ${type}: ${pred_actual / word_count}")

    print(confusion_matrix)

if __name__ == "__main__":
    main()
