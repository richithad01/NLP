#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import gensim.downloader as api


# In[3]:


# --- NLTK Setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[4]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[5]:


df = pd.read_csv("IMDB Dataset.csv")  # ✅ ADDED YOUR DOWNLOADS PATH
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # remove HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove numbers/symbols
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

df['clean_review'] = df['review'].apply(clean_text)


# In[6]:


# --- 2. TF-IDF + Logistic Regression ---
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean_review'])
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['sentiment'], test_size=0.2, random_state=42)

model1 = LogisticRegression()
model1.fit(X_train, y_train)
preds1 = model1.predict(X_test)

print("1️⃣ TF-IDF + Logistic Regression")
print("Accuracy:", accuracy_score(y_test, preds1))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds1))


# In[7]:


#task 1 introduction and importing the data
#task 2 transforming documents into feature vector data
# task TDIDF
#task 4 data prepartation 
#task 6 tokenixation od documents
#task 7 document classification using logistic regression
# task u load daved model grom disk
#task 9  model accuracy

import pandas as pd
df = pd.read_csv('IMDB Dataset.csv')
df.head(10)


# In[8]:


import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
doc = np.array(['The sun is shinhin',
                'The weather is sweet'
                'and one and one is two'])
bag  = count.fit_transform(doc)


# In[9]:


print(count.vocabulary_)


# In[10]:


print(bag.toarray())


# In[ ]:




