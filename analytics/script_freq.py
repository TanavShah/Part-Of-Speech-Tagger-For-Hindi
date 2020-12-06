import csv

di = dict()
with open('../dataset/naive_bayes/dataset.csv') as f1:
    reader = csv.reader(f1)
    for row in reader:
      if row[0] in di:
         di[row[0]] = di[row[0]] + 1
      else:
         di[row[0]] = 1

with open('word_tag_frequency.csv', 'w') as f2:
    writer = csv.writer(f2)
    for key, value in di.items():
            new_row = [key, value]
            writer.writerow(new_row)            