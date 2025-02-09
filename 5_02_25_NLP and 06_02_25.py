#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nltk.stem import PorterStemmer
stemmerporter = PorterStemmer()
stemmerporter.stem('happiness')


# In[3]:


from nltk.stem import LancasterStemmer
stemmerporter = LancasterStemmer()
stemmerporter.stem('happiness')


# In[5]:


from nltk.stem import RegexpStemmer
stemmerporter = RegexpStemmer('ing')
stemmerporter.stem('singing')


# In[6]:


from nltk.stem import LancasterStemmer
stemmerporter = LancasterStemmer()
stemmerporter.stem('singing')


# In[7]:


from nltk.stem import PorterStemmer
stemmerporter = PorterStemmer()
stemmerporter.stem('singing')


# In[8]:


from nltk.stem import SnowballStemmer
SnowballStemmer.languages
frenchstemmer = SnowballStemmer('french')
frenchstemmer.stem('manges')


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer


# In[18]:


vect = CountVectorizer(binary = True)
corpus = ['Tessaract is good optical character recognition engine','optical character recognition is significant']
vect.fit(corpus)#fit function studies corpus and pulls unique words

vocab = vect.vocabulary_ #attribute
for key in sorted(vocab.keys()):
    print("{}:{}".format(key,vocab[key]))

print(vect.transform(['This is a good optical illusion. It is a good one.']))

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vect.transform(['Google Cloud is a good optical character ']).toarray())
print(similarity)

print(vect.transform(['OCR is a character recognition engine']))


# In[62]:


def gender_features(word):
    return {'last_letter' : word[-1]}


# In[63]:


gender_features('Kirti')


# In[34]:


import nltk
nltk.download('names')
from nltk.corpus import names


# In[35]:


names.words()
print(len(names.words()))


# In[32]:


labeled_names = ([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')] )


# In[33]:


import random
random.shuffle(labeled_names)


# In[38]:


featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]


# In[52]:


train_set,test_set = featuresets[5000:],featuresets[:5000]


# In[53]:


import nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[54]:


classifier.classify(gender_features('Richitha'))


# In[55]:


print(nltk.classify.accuracy(classifier,test_set))


# In[66]:


import nltk
f=open('tweets1.txt','r')


# In[67]:


text = f.read()
text1 = text.split()
text2 = nltk.Text(text1)


# In[70]:


text2.concordance("the")


# In[1]:


from urllib import request


# In[2]:


url = "https://www.gutenberg.org/cache/epub/75320/pg75320.txt"


# In[3]:


response = request.urlopen(url)


# In[4]:


raw = response.read().decode('utf8')


# In[5]:


type(raw)


# In[6]:


len(raw)


# In[7]:


raw[:75]


# In[14]:


import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
tokens = word_tokenize(raw)


# In[15]:


type(tokens)


# In[ ]:




