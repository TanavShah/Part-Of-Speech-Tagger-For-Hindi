import pyconll
from pathlib import Path
import csv
from tqdm import tqdm

UD_HINDI_TRAIN = Path(__file__).parent / './hi_hdtb-ud-train.conllu'
UD_HINDI_TEST = Path(__file__).parent / './hi_hdtb-ud-test.conllu'
UD_HINDI_DEV = Path(__file__).parent / './hi_hdtb-ud-dev.conllu'


def generate_csv(file, name):
    data = pyconll.load_from_file(file)

    word_pos = list()

    for sentence in tqdm(data):
        for token in sentence:   
            word : str = str(token.form)
            tag : str = str(token.xpos) 
            word_pos.append([word, tag])
        word_pos.append([" ", " "])
        
    with open(f'dataset/stemming/{name}_set.csv','w') as out:
        csv_out=csv.writer(out, delimiter="~")
        csv_out.writerow(['tokens','tags'])
        for row in word_pos:
            csv_out.writerow(row)    

def main():
    generate_csv(UD_HINDI_TRAIN, 'train')
    generate_csv(UD_HINDI_TEST, 'test')
    generate_csv(UD_HINDI_DEV, 'dev')

if __name__ == "__main__":
    main()
