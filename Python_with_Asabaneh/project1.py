def find_most_common_words(filename, x):
    
    ''' This function will take two inputs, 
    one is sample text to look for most common words 
    and the next one is the number of most common words'''

    import re
    pattern = r'[a-zA-Z]+'
    with open(filename,'rb') as f:
        a_file = f.read().decode(errors='replace')
        word_list = re.findall(pattern,a_file)
        

    word_count = {}
    for word in word_list:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    result = sorted(word_count.items(), key=lambda x:x[1],reverse=True)
    return result[0:x]


print(find_most_common_words('sample_text.txt',10))
