import csv

with open('token-tag_pairs_cleaned.csv') as csv_file:
    reader = csv.reader(csv_file)
    with open('dataset.csv', 'w') as g:
        writer = csv.writer(g)
        for row in reader:
            new_row = ['_'.join([row[0], row[1]])]
            writer.writerow(new_row)
