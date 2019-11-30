#!/usr/bin/env python
# coding: utf-8

# In[166]:


import pandas as pd



# In[167]:


df=pd.read_csv('iphone6-negative.csv',encoding='latin-1')

stop_words=open('stopwords.txt','r').read()
df.head()


# In[168]:


df=df[['Text','Sentiment']]


# In[169]:


def denoise(text):
    return re.sub('[^a-zA-Z]*',',text').strip()


# In[170]:


def stopwords_remove(text):
    text=text.lower()
    words=text.split(' ')
    clean_words=[]
    for word in words:
        if word in stop_words:
            pass 
        else:
            clean_words.append(word)
    text=' '.join(clean_words) 
    return text


# In[171]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))


# In[172]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()
import nltk
def stemm(text):
    clean_text=''
    for word in text.split():
        clean_text +=" "+ ps.stem("word")
        return text 


# In[173]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))


# In[174]:


#text='fuck the iPhone 6s cus Im not getting one :'


# In[175]:


def mod (text):
    denoise(text)
    stopwords_remove(text)
    stemm(text)
    return text 


# In[176]:


text


# In[177]:


from sklearn.feature_extraction.text import TfidfVectorizer
Tf_idf=TfidfVectorizer()
clean_text=Tf_idf.fit_transform(df.Text).toarray()


# In[ ]:





# In[178]:


df.head()


# In[ ]:





# In[179]:


df.Text.apply(stopwords_remove)          ##to applay afunction on all data


# In[180]:


from sklearn.feature_extraction.text import CountVectorizer


# In[181]:


cv = CountVectorizer()


# In[182]:


x = cv.fit_transform(df.Text)


# In[183]:


from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
y=le.fit_transform(df.Sentiment)


# In[184]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[185]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))


# In[ ]:





# In[ ]:




