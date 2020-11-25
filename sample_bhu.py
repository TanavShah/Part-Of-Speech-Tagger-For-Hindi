  
#DO NOT CHANGE!
def read_train_file():
  '''
  HELPER function: reads the training files containing the words and corresponding tags.
  Output: A tuple containing 'sentences' and 'tags'
  'sentences': It is a list of sentences where each sentence, in turn, is a list of words.
  For example - [['A','boy','is','running'],['Pick','the','red','cube'],['One','ring','to','rule','them','all']]
  'tags': A nested list similar to above, just the corresponding tags instead of words. 
  '''           
  f = open('train_data.txt','r')
  sentences = []
  tags = []
  sentence = []
  tag = []
  for line in f:
    s = line.rstrip('\n')
    if s == '':
      sentences.append(sentence)
      tags.append(tag)
      sentence=[]
      tag=[]
    else:
      w,t = line.split()
      sentence.append(w)
      tag.append(t)
  sentences = sentences[1:]
  tags = tags[1:]
  assert len(sentences) == len(tags)
  f.close()
  return (sentences,tags)








#NEEDS TO BE FILLED!
def store_emission_and_transition_probabilities(train_list_words, train_list_tags):
    
    '''
  This creates dictionaries storing the transition and emission probabilities - required for running Viterbi. 
  INPUT: The nested list of words and corresponding nested list of tags from the TRAINING set. This passing of correct lists and calling the function
  has been done for you. You only need to write the code for filling in the below dictionaries. (created with bigram-HMM in mind)
  OUTPUT: The two dictionaries

  HINT: Keep in mind the boundary case of the starting POS tag. You may have to choose (and stick with) some starting POS tag to compute bigram probabilities
  for the first actual POS tag.
    '''

    tag_follow_tag = {}    
    
    '''Nested dictionary to store the transition probabilities
    each tag X is a key of the outer dictionary with an inner dictionary as the corresponding value
    The inner dictionary's key is the tag Y following X
    and the corresponding value is the number of times Y follows X - convert this count to probabilities finally before returning 
    for example - { X: {Y:0.33, Z:0.25}, A: {B:0.443, W:0.5, E:0.01}} (and so on) where X,Y,Z,A,B,W,E are all POS tags
    so the first key-dictionary pair can be interpreted as "there is a probability of 0.33 that tag Y follows tag X, and 0.25 probability that Z follows X"
    '''
    word_tag = {}
    """Nested dictionary to store the emission probabilities.
  Each word W is a key of the outer dictionary with an inner dictionary as the corresponding value
  The inner dictionary's key is the tag X of the word W
  and the corresponding value is the number of times X is a tag of W - convert this count to probabilities finally before returning
  for example - { He: {A:0.33, N:0.15}, worked: {B:0.225, A:0.5}, hard: {A:0.1333, W:0.345, E:0.25}} (and so on) where A,N,B,W,E are all POS tags
  so the first key-dictionary pair can be interpreted as "there is a probability of 0.33 that A is the POS tag for He, and 0.15 probability that N is the POS tag for He"
  """


  # *** WRITE YOUR CODE HERE ***    

  # tag_follow_tag

    total = 0
    labels = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.', '*']
    count_labels = {}
    for i in range(len(labels)):
        count_labels[labels[i]] = 0
        #print(labels[i], end = " ")

    for i in range(len(labels)):
      tag_follow_tag[labels[i]] = {}
      for j in range(len(labels)):
        tag_follow_tag[labels[i]][labels[j]] = 0

    last = {}
    for i in range(len(labels)):
      last[labels[i]] = 0

    for tags_seq in train_list_tags:
      tag_follow_tag['*'][tags_seq[0]] += 1
      count_labels[tags_seq[len(tags_seq) - 1]] += 1
      last[tags_seq[len(tags_seq) - 1]] += 1
      for i in range(len(tags_seq) - 1):
        tag_follow_tag[tags_seq[i]][tags_seq[i + 1]] += 1
        count_labels[tags_seq[i]] += 1

    for i in range(len(labels)):
      for j in range(len(labels)):
        if (count_labels[labels[i]] == 0): 
          continue

        tag_follow_tag[labels[i]][labels[j]] /= count_labels[labels[i]]   


    for i, j in tag_follow_tag['*'].items():
      tag_follow_tag['*'][i] /= len(train_list_tags)

    for i in range(len(labels)):
      if count_labels[labels[i]] == 0:
        continue
      last[labels[i]] /= count_labels[labels[i]]

    for i, j in last.items():
      tag_follow_tag[i]['*'] = j
    

    # word_tag

    word_count = {}
    for i in range(len(train_list_words)):
      for j in range(len(train_list_words[i])):
        if train_list_words[i][j] not in word_tag.keys():
          word_tag[train_list_words[i][j]] = {train_list_tags[i][j] : 1}
          word_count[train_list_words[i][j]] = 0
          for k in range(len(labels)):   
            if labels[k] == train_list_tags[i][j]:
              continue
            word_tag[train_list_words[i][j]][labels[k]] = 0
        else:
          word_tag[train_list_words[i][j]][train_list_tags[i][j]] += 1

    for i, j in word_tag.items():
      for k, l in j.items():
        word_count[i] += l

    for i, j in word_tag.items():
      for k, l in j.items():              
        word_tag[i][k] /= word_count[i]


    # count = 0
    # for i, j in word_tag.items():
    #   count = 0
    #   for k, l in j.items():
    #     count += l

    #   print(count)

  
    # END OF YOUR CODE  

    return (tag_follow_tag, word_tag)



#NEEDS TO BE FILLED!
def assign_POS_tags(test_words, tag_follow_tag, word_tag):

    '''
  This is where you write the actual code for Viterbi algorithm. 
  INPUT: test_words - this is a nested list of words for the TEST set
         tag_follow_tag - the transition probabilities (bigram), filled in by YOUR code in the store_emission_and_transition_probabilities
         word_tag - the emission probabilities (bigram), filled in by YOUR code in the store_emission_and_transition_probabilities
  OUTPUT: a nested list of predicted tags corresponding to the input list test_words. This is the 'output_test_tags' list created below, and returned after your code
  ends.

  HINT: Keep in mind the boundary case of the starting POS tag. You will have to use the tag you created in the previous function here, to get the
  transition probabilities for the first tag of sentence...
  HINT: You need not apply sophisticated smoothing techniques for this particular assignment.
  If you cannot find a word in the test set with probabilities in the training set, simply tag it as 'NOUN'. 
  So if you are unable to generate a tag for some word due to unavailibity of probabilities from the training set,
  just predict 'NOUN' for that word.

    '''



    output_test_tags = []    #list of list of predicted tags, corresponding to the list of list of words in Test set (test_words input to this function)


    # *** WRITE YOUR CODE HERE *** 

    labels = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.', '*']
    dp1, dp2 = {}, {}
    for i in range(len(test_words)):
      output_test_tags.append([])
      n = len(test_words[i])        
      # dp1[0] = {} 
      # dp2[0] = {}         
      # dp1[0]['*'] = 1
      # for j in range(len(labels)):
      #     if labels[j] != '*':
      #         dp1[0][labels[j]] = 0

      dp1[1], dp2[1] = {}, {}
      for j in range(len(labels)):
        temp = 0
        if test_words[i][0] in word_tag.keys():
          if tag_follow_tag['*'][labels[j]] * word_tag[test_words[i][0]][labels[j]] > temp:
            temp = tag_follow_tag['*'][labels[j]] * word_tag[test_words[i][0]][labels[j]]

        dp1[1][labels[j]] = temp
        dp2[1][labels[j]] = '*'

      
      output_test_tags[i].append('NOUN')
      for j in range(1, n):
          output_test_tags[i].append('NOUN')
          dp1[j + 1] = {}
          dp2[j + 1] = {}
          for label in labels:
              dp1[j + 1][label] = 0
              dp2[j + 1][label] = 'NOUN'

      for j in range(2, n + 1):
        if test_words[i][j - 1] not in word_tag.keys():
          dp1[j]['NOUN'] = 1      
          for k in range(len(labels)):
            temp = 0
            for l in range(len(labels)):
              if dp1[j - 1][labels[l]] * tag_follow_tag[labels[l]][labels[k]] > temp:
                temp = dp1[j - 1][labels[l]] * tag_follow_tag[labels[l]][labels[k]]
                dp2[j][labels[k]] = labels[l]

        else:
          for k in range(len(labels)):
            temp, temp2 = 0, 0          
            for l in range(len(labels)):
              temp = dp1[j - 1][labels[l]] * tag_follow_tag[labels[l]][labels[k]] * word_tag[test_words[i][j - 1]][labels[k]]
              if temp > dp1[j][labels[k]]:
                dp1[j][labels[k]] = temp

              if dp1[j - 1][labels[l]] * tag_follow_tag[labels[l]][labels[k]] > temp2:
                temp2 = dp1[j - 1][labels[l]] * tag_follow_tag[labels[l]][labels[k]]
                dp2[j][labels[k]] = labels[l]

      # temp = 0
      # dp2[n + 1] = {}
      # for j in range(len(labels)):
      #   if dp1[n][labels[j]] * tag_follow_tag[labels[j]]['*'] > temp:
      #     temp = dp1[n][labels[j]] * tag_follow_tag[labels[j]]['*']
      #     dp2[n + 1]['*'] = labels[j]
      #   else:
      #     dp2[n + 1]['*'] = 'NOUN'

      temp = 0
      for j in range(len(labels)):
        if dp1[n][labels[j]] > temp:
          temp = dp1[n][labels[j]]
          output_test_tags[i][n - 1] = labels[j]

      curr = output_test_tags[i][n - 1]
      pos = n - 1
      while pos != -1:
        output_test_tags[i][pos] = curr
        pos -= 1
        curr = dp2[pos + 2][curr]



      # for j in range(1, n + 1):
      #   temp = 0
      #   for k in range(len(labels)):
      #     if dp1[j][labels[k]] > temp:
      #       temp = dp1[j][labels[k]]
      #       output_test_tags[i][j - 1] = labels[k]

      # for j in range(2, n + 1):
      #   temp = 0
      #   for k in range(len(labels)):
      #     if dp[j][k] > temp:
      #       temp = dp[j][k]
      #       temp2 = 0
      #       for l in range(len(labels)):
      #         if tag_follow_tag[labels[k]][labels[l]] > temp2:
      #           temp2 = tag_follow_tag[labels[k]][labels[l]]
      #           output_test_tags[j - 2] = l

    # END OF YOUR CODE

    return output_test_tags









# DO NOT CHANGE!
def public_test(predicted_tags):
  '''
  HELPER function: Takes in the nested list of predicted tags on test set (prodcuced by the assign_POS_tags function above)
  and computes accuracy on the public test set. Note that this accuracy is just for you to gauge the correctness of your code.
  Actual performance will be judged on the full test set by the TAs, using the output file generated when your code runs successfully.
  '''

  f = open('public_test_data.txt','r')
  sentences = []
  tags = []
  sentence = []
  tag = []
  for line in f:
    s = line.rstrip('\n')
    if s == '':
      sentences.append(sentence)
      tags.append(tag)
      sentence=[]
      tag=[]
    else:
      w,t = line.split()
      sentence.append(w)
      tag.append(t)
  sentences = sentences[1:]
  tags = tags[1:]
  assert len(sentences) == len(tags)
  f.close()
  public_predictions = predicted_tags[:len(tags)]
  assert len(public_predictions)==len(tags)

  flattened_actual_tags = []
  flattened_pred_tags = []
  for i in range(len(tags)):
    x = tags[i]
    y = public_predictions[i]
    flattened_actual_tags+=x
    flattened_pred_tags+=y
  assert len(flattened_actual_tags)==len(flattened_pred_tags)

  correct = 0.0
  for i in range(len(flattened_pred_tags)):
    if flattened_pred_tags[i]==flattened_actual_tags[i]:
      correct+=1.0
  print('Accuracy on the Public set = '+str(correct/len(flattened_pred_tags)))



# DO NOT CHANGE!
if __name__ == "__main__":
  words_list_train = read_train_file()[0]
  tags_list_train = read_train_file()[1]

  dict2_tag_tag = store_emission_and_transition_probabilities(words_list_train,tags_list_train)[0]
  word_tag = store_emission_and_transition_probabilities(words_list_train,tags_list_train)[1]

  f = open('private_unlabelled_test_data.txt','r')

  words = []
  l=[]
  for line in f:
    w = line.rstrip('\n')
    if w=='':
      words.append(l)
      l=[]
    else:
      l.append(w)
  f.close()
  words = words[1:]
  test_tags = assign_POS_tags(words, dict2_tag_tag, word_tag)
  assert len(words)==len(test_tags)

  public_test(test_tags)

  #create output file with all tag predictions on the full test set

  f = open('output.txt','w')
  for i in range(len(words)):
    sent = words[i]
    pred_tags = test_tags[i]
    for j in range(len(sent)):
      word = sent[j]
      pred_tag = pred_tags[j]
      f.write(word+' '+pred_tag)
      f.write('\n')
    f.write('\n')
  f.close()

  print('OUTPUT file has been created') 