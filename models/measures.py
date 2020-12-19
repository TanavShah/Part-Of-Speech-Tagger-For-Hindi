data = [["DEM  :  0.9501084598698482"],
["NNP  :  0.747364722417428"],
["PSP  :  0.978975215817321"],
["INTF  :  0.8846153846153846"],
["JJ  :  0.827211588204863"],
["NN  :  0.8383233532934131"],
["QC  :  0.8330522765598651"],
["VM  :  0.9017293439472962"],
["SYM  :  0.9917355371900827"],
["PRP  :  0.9443620178041543"],
["NNC  :  0.5851938895417156"],
["VAUX  :  0.9833032490974729"],
["CC  :  0.9868114817688131"],
["NNPC  :  0.7412060301507538"],
["NST  :  0.968"],
["RP  :  0.9034907597535934"],
["NSTC  :  1.0"],
["QF  :  0.9285714285714286"],
["JJC  :  0.3125"],
["WQ  :  0.9523809523809523"],
["NEG  :  0.9894736842105263"],
["RB  :  0.7185185185185186"],
["RDP  :  0.5"],
["RBC  :  0"],
["QO  :  0.7368421052631579"],
["CCC  :  1.0"],
["QCC  :  0.9797979797979798"],
["UNK  :  0"],
["PRPC  :  0"],
["QFC  :  0"],
["INJ  :  0"],]

inner_p_tag = [e[0].split()[0] for e in data]

cm = [[442, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2047, 0, 0, 22, 179, 8, 11, 0, 0, 14, 0, 0, 105, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 457, 0, 0, 0], [0, 0, 7091, 0, 9, 64, 0, 5, 0, 0, 0, 2, 3, 0, 0, 7, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 3, 1, 1773, 16, 0, 1, 0, 1, 16, 0, 2, 13, 1, 0, 0, 2, 3, 0, 0, 0, 0, 1, 0, 0, 0, 98, 0, 0, 0], [0, 169, 9, 0, 49, 6480, 4, 10, 0, 0, 80, 0, 0, 49, 1, 5, 0, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 320, 0, 0, 0], [0, 5, 0, 0, 0, 0, 519, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 59, 0, 0, 0], [0, 1, 7, 0, 7, 9, 1, 3467, 0, 0, 0, 118, 0, 2, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 26, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2420, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [39, 0, 0, 0, 0, 0, 0, 1, 0, 1290, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 26, 0, 0, 32, 122, 0, 3, 0, 0, 535, 0, 3, 61, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 64, 1, 0, 0], [0, 0, 2, 0, 0, 0, 1, 32, 0, 0, 0, 2179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1277, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 148, 10, 0, 37, 62, 12, 3, 0, 0, 77, 1, 3, 1030, 0, 0, 0, 2, 3, 0, 0, 1, 0, 0, 1, 0, 0, 201, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 488, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 0], [0, 9, 10, 0, 0, 0, 0, 0, 0, 2, 1, 0, 11, 1, 0, 451, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 0, 0, 0, 0, 0, 1, 1, 0, 5, 0, 0, 0, 0, 268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 7, 10, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 4, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 111, 0, 1, 0, 0, 0, 11, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 10, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 23, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

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

macro_precision = total_precision/31
macro_recall = total_recall/31
macro_f_score = ((2*macro_precision*macro_recall)/(macro_precision + macro_recall))

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