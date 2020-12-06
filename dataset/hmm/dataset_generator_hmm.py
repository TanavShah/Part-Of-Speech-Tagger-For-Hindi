import csv

d1 = dict()
d2 = dict()
d3 = dict()

i = 0

with open('../../analytics/token-tag_pairs_cleaned.csv') as f1:
    reader = csv.reader(f1)
    for row in reader:
        if(i==0):
            i+=1
            continue
        d1[i] = [row[0], row[1]]
        i = i+1
       

i = 0

print("tokens~tags")

with open('test_set_indices_hmm.csv') as f2:
    reader = csv.reader(f2)
    for row in reader:
        if(i==0):
            i+=1
            continue
        for j in range(int(row[0]), int(row[1]) + 1):
            d2[i-1] = [d1[j][0], d1[j][1]]
            st = d2[i - 1][0] + "~" + d2[i - 1][1]
            # print (st)
            # print (d2[i - 1][0],d2[i - 1][1])    
            i+=1
        
        d2[i] = ["~","~"]
        # print (d2[i][0], "," , d2[i][1])
        # print(" ~ ")
        i = i+1 

i = 0

# print("tokens,tags")


with open('train_set_indices_hmm.csv') as f3:
    reader = csv.reader(f3)
    for row in reader:
        if(i==0):
            i+=1
            continue
        if(row[0] == "0") :
            continue
        # print (row[0],row[1])
        for j in range(int(row[0]), int(row[1]) + 1):
            d3[i-1] = [d1[j][0], d1[j][1]]
            st = d3[i - 1][0] + "~" + d3[i - 1][1]
            print (st)  
            i+=1
        
        d3[i] = ["~","~"]
        print(" ~ ")
        i = i+1 

