#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nltk


# In[2]:


import nltk


# In[3]:


nltk.download()


# In[4]:


from nltk.book import *


# In[5]:


nltk.download('gutenberg')


# In[6]:


from nltk.book import *


# In[7]:


nltk.download('genesis')


# In[8]:


from nltk.book import *


# In[9]:


nltk.download('inaugural')


# In[11]:


nltk.download('nps_chat')


# In[13]:


nltk.download('webtext')


# In[15]:


nltk.download('treebank')


# In[16]:


from nltk.book import *


# In[20]:


from nltk.corpus import brown
nltk.download('brown')


# In[21]:


brown.categories()


# In[23]:


brown.words(categories = 'adventure')[:50]


# In[24]:


brown.words(categories = 'humor')[:10]


# In[25]:


from nltk.corpus import inaugural


# In[26]:


inaugural.fileids()


# In[ ]:




