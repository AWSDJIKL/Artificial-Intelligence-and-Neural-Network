import numpy as np

def getdata(datanum):
    text = open('mytext.txt','rb').read().lower()
    print('len of text:%d'%len(text))
    
    maxlen=60
    step=3
    sen=[]
    nextchar=[]
    
    for i in range(0,len(text)-maxlen,step):
        sen.append(text[i:i+maxlen])
        nextchar.append(text[i+maxlen])
    print('number of sentences: %d'%len(sen))
    
    ##char to number
    chars = sorted(list(set(text)))
    char_indices = dict((char, chars.index(char)) for char in chars)
    
    if datanum==2:
        ##one hot
        x = np.zeros((len(sen), maxlen))
        y = np.zeros((len(sen)))
        for i, isen in enumerate(sen):
            for t, char in enumerate(isen):
                x[i, t]=char_indices[char]
            y[i]=char_indices[nextchar[i]]
        return char_indices,x,y
    else:
        return char_indices

