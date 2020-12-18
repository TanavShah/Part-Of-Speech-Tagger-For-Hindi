import csv
from pathlib import Path
from collections import Counter

d1 = dict()
d2 = dict()
train_file_name = Path(__file__).parent / '../dataset/stemming/train_set.csv'

with open(train_file_name) as f1:
    reader = csv.reader(f1, delimiter = '~')
    for row in reader:
        if (row[0]==" "):
            continue
        if row[1] in d2:
            d2[row[1]] = d2[row[1]] + 1
        elif row[1] not in d2:
            d2[row[1]] = 1
        if row[0] in d1:
            d1[row[0]] = d1[row[0]] + 1
        elif row[0] not in d1:
            d1[row[0]] = 1

# k1 = Counter(d1)
# h1 = k1.most_common()

# with open('word_frequency_stem.csv', 'w') as f2:
#     for i in h1:
#         f2.write('%s~%s\n' % (i[0], i[1]))

k2 = Counter(d2)
h2 = k2.most_common()

for i,j in h2 :
    print(i, " : ", j)

# with open('tags_frequency_stem.csv', 'w') as f3:
#     for j in h2:
#         f3.write('%s,%s\n' % (j[0], j[1]))