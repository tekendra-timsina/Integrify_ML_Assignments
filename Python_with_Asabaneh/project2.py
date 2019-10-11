def compare_similarity(textfile1,textfile2):
    ''' This function will compare the two string files and find out the similarity.
    It will only check the valuable words similarity excluding common words such as pronoun, conjuction, verb'''

    # reading and processing the first file
    with open(textfile1,'r') as f:
        file1 = f.read()
        word_list1 = file1.split()

    important_word1 = []
    for word in word_list1:
        if len(word) > 4:
            word = word.strip('.')
            important_word1.append(word)
    print("The word list from first text file after removing grammatical word (eg. pronoun,conjunction) is of length: \n {}".format(len(important_word1)))
    # reading and processing the second file
    with open(textfile2,'r') as f:
        file2 = f.read()
        word_list2 = file2.split()

    important_word2 = []
    for word in word_list2:
        if len(word) > 4:
            word = word.strip('.')
            important_word2.append(word)
    print("The word list from second text file after removing grammatical word (eg. pronoun,conjunction) is of length: \n {}".format(len(important_word2)))

    # Comparing the words to find common words
    common_words = []
    for word in important_word1:
        if word not in common_words and word in important_word2:
            common_words.append(word)
    print("The common word list length is \n {}".format(len(common_words)))
    print("The percentage of common words to the first speech is \n {} %".format(round(len(common_words)/len(important_word1)*100,2)))
    print("\n")
    
    return f'Words that are common are:\n {common_words}'

print(compare_similarity('melina_trump_speech.txt','michelle_obama_speech.txt'))
