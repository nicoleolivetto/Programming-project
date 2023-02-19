#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings


# In[2]:


warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')


# In[3]:


topsubscribed_df =pd.read_csv("topSubscribed.csv")
topsubscribed_df


# In[4]:


topsubscribed_df.dtypes


# In[5]:


pd.isnull(topsubscribed_df)


# In[6]:


topsubscribed_df['Subscribers']=topsubscribed_df['Subscribers'].str.replace(',','')
topsubscribed_df['Video Views']=topsubscribed_df['Video Views'].str.replace(',','')
topsubscribed_df['Video Count']=topsubscribed_df['Video Count'].str.replace(',','')


# In[7]:


pd.isnull(topsubscribed_df).sum()


# In[8]:


topsubscribed_df['Subscribers'] = topsubscribed_df['Subscribers'].astype('int64')
topsubscribed_df['Video Views'] = topsubscribed_df['Video Views'].astype('int64')
topsubscribed_df['Video Count'] = topsubscribed_df['Video Count'].astype('int64')


# In[9]:


topsubscribed_df=topsubscribed_df[(topsubscribed_df != 0).all(1)]


# In[10]:


topsubscribed_df


# In[11]:


topsubscribed_df.dtypes


# In[12]:


topsubscribed_df.head()


# In[13]:


topsubscribed_df.tail()


# In[14]:


topsubscribed_df.shape


# In[15]:


topsubscribed_df.info()


# In[16]:


topsubscribed_df.describe()


# In[17]:


topsubscribed_df.duplicated().sum()


# In[18]:


topsubscribed_df.reset_index(drop=True, inplace=True)


# In[19]:


#topsubscribed_df["range_subs"] 
topsubscribed_df=topsubscribed_df.assign(range_subs=3)
topsubscribed_df.range_subs = topsubscribed_df.range_subs.astype(str)


# In[20]:


for i in range ((len(topsubscribed_df["Subscribers"]))):
    if (topsubscribed_df.at[(i),'Subscribers'] > 10000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 15000000):
        topsubscribed_df.at[(i),'range_subs']= '10M - 15M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 15000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 20000000):
        topsubscribed_df.at[(i),'range_subs']= '15M - 20M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 20000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 50000000):
        topsubscribed_df.at[(i),'range_subs']= '20M - 50M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 50000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 100000000):
        topsubscribed_df.at[(i),'range_subs']= '50M - 100M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 100000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 200000000):
        topsubscribed_df.at[(i),'range_subs']= '100M - 200M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 20000000):
        topsubscribed_df.at[(i),'range_subs']= '200M+'

    


# In[21]:


topsubscribed_df = topsubscribed_df.drop(topsubscribed_df[topsubscribed_df['Started'] < 2005].index)


# In[22]:


topsubscribed_df


# In[23]:


topsubscribed_df.groupby(['range_subs']).size().sort_values(ascending = False)


# In[24]:


#topsubscribed_df['range_subs'].value_counts().sort_index().plot.bar(x='Target Value', y='Number of Occurrences')
#plt.rcParams["figure.figsize"] = (10, 5)

plt.figure(figsize=(15,10))
plt.title('Different Manifacturing companies')
sns.set_style('whitegrid')
m = sns.barplot(x=topsubscribed_df['range_subs'].value_counts().index, y=topsubscribed_df['range_subs'].value_counts(), palette ='Paired')
m.set_xticklabels(topsubscribed_df['range_subs'].value_counts().index, rotation = 45)
m.set(xlabel ='range_subs', ylabel = 'Values')
plt.show()


# In[25]:


pie_chart = topsubscribed_df['range_subs'].value_counts()
fuel_pie = pie_chart.plot.pie(shadow=True,startangle=90,autopct='%1.1f%%')
plt.legend(topsubscribed_df['range_subs'])
plt.show()


# In[26]:


#analysis on  car production
annual_join = topsubscribed_df.groupby(['Started']).size().reset_index().rename(columns = {0:'Counts'})
annual_join


# In[27]:


#topsubscribed_df['Started'].value_counts().sort_index().plot.bar(x='Target Value', y='Number of Occurrences')
#plt.rcParams["figure.figsize"] = (10, 5)

plt.figure(figsize=(15,10))
plt.title('Different Manifacturing companies')
sns.set_style('whitegrid')
a = sns.barplot(x=topsubscribed_df['Started'].value_counts().sort_index().index, y=topsubscribed_df['Started'].value_counts(), palette ='Paired')
a.set_xticklabels(topsubscribed_df['Started'].value_counts().index, rotation = 45)
a.set(xlabel ='Started', ylabel = 'Values')
plt.show()


# In[28]:


topsubscribed_df['Started'].unique()


# In[29]:


#topsubscribed_df.groupby(['Category']).size().sort_values(ascending = False)
Cat = topsubscribed_df.groupby(['Category']).size().reset_index().rename(columns = {0:'Counts'})
Cat   


# In[30]:


#topsubscribed_df['Category'].value_counts().sort_index().plot.bar(x='Target Value', y='Number of Occurrences')
#plt.rcParams["figure.figsize"] = (10, 5)

plt.figure(figsize=(15,10))
plt.title('Different Manifacturing companies')
sns.set_style('whitegrid')
n = sns.barplot(x=topsubscribed_df['Category'].value_counts().sort_index().index, y=topsubscribed_df['Category'].value_counts(), palette ='Paired')
n.set_xticklabels(topsubscribed_df['Category'].value_counts().index, rotation = 90)
n.set(xlabel ='Started', ylabel = 'Values')
plt.show()


# In[31]:


topsubscribed_df


# In[32]:


#stampo la media di videocount per pgni sub range
print( topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Count'].transform('mean').unique())
fig, ax = plt.subplots()
#rappresesnto  come cambia la media di video count a seconda del sub range
ax.plot(topsubscribed_df['range_subs'].unique(), topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Count'].transform('mean').unique())
ax.set_xlabel("range")
ax.set_ylabel("media video count")
plt.show()


# In[33]:


#stampo la media di video views per pgni sub range
print( topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Views'].transform('mean').unique())
fig, ax = plt.subplots()
#rappresesnto  come cambia la media di video views a seconda del sub range
ax.plot(topsubscribed_df['range_subs'].unique(), topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Views'].transform('mean').unique())
ax.set_xlabel("range")
ax.set_ylabel("media video views")
plt.show()


# In[34]:


topsubscribed_df.describe().T


# In[35]:


topsubscribed_df.hist(figsize=(12,10), bins=50)


# In[36]:


sns.set(style="ticks", color_codes=True)    
sns.pairplot(topsubscribed_df)
plt.show()


# In[37]:


#plt.title('Comparison between MS and Apple')
plt.scatter(topsubscribed_df['Started'], topsubscribed_df['Subscribers'])
plt.xlabel('Started')
plt.ylabel('Subscribers')
plt.grid()
plt.show()


# In[38]:


plt.scatter(topsubscribed_df['Video Views'], topsubscribed_df['Subscribers'])
plt.xlabel('Video Views')
plt.ylabel('Subscribers')
plt.grid()
plt.show()


# In[39]:


plt.scatter(topsubscribed_df['Video Count'], topsubscribed_df['Started'])
plt.xlabel('Video count')
plt.ylabel('started')
plt.grid()
plt.show()


# In[40]:


import seaborn as sb
plt.figure(figsize=(8,6))
sb.heatmap(topsubscribed_df.corr(), annot=True)


# In[ ]:





# In[41]:


#come il video count cambia a seconda del range subs
plt.figure(figsize=(25,15))
sns.lineplot(x = 'range_subs', y = 'Video Count', data=topsubscribed_df)
plt.title('Prodcution year')
plt.grid()
plt.show()


# In[42]:


#come il video views cambia a seconda del range subs
plt.figure(figsize=(25,15))
sns.lineplot(x = 'range_subs', y = 'Video Views', data=topsubscribed_df)
plt.title('come il video views cambia a seconda del range subs')
plt.grid()
plt.show()


# In[43]:


#come il range subs influisce su video views e su video count

plt.figure(figsize=(30,10))
plt.subplot(121)

plt.title('Video views depending on range subs',fontsize = 20)
plt.ylabel("video views")
topsubscribed_df.groupby('range_subs')['Video Views'].mean().sort_index().plot.bar(color = 'pink')
plt.grid()
plt.subplot(122)
plt.title('Video count depending on range subs',fontsize = 20)
plt.ylabel("video count")
topsubscribed_df.groupby('range_subs')['Video Count'].mean().sort_index().plot.bar(color = 'yellow')
plt.grid()
plt.show()


# In[44]:


r1=(topsubscribed_df[topsubscribed_df['range_subs']=='10M - 15M'])
r2=(topsubscribed_df[topsubscribed_df['range_subs']=='15M - 20M'])
r3=(topsubscribed_df[topsubscribed_df['range_subs']=='20M - 50M'])
r4=(topsubscribed_df[topsubscribed_df['range_subs']=='50M - 100M'])
r5=(topsubscribed_df[topsubscribed_df['range_subs']=='100M - 200M'])
r6=(topsubscribed_df[topsubscribed_df['range_subs']=='200M+'])

r2


# In[45]:


fig, axes = plt.subplots(2,3, figsize=(25, 15))

sb.heatmap(r1.corr(), annot=True,ax=axes[0,0])
sb.heatmap(r2.corr(), annot=True,ax=axes[0,1])
sb.heatmap( r3.corr(), annot=True,ax=axes[0,2])
sb.heatmap(r4.corr(), annot=True,ax=axes[1,0])
sb.heatmap( r5.corr(), annot=True,ax=axes[1,1])
sb.heatmap(r6.corr(), annot=True,ax=axes[1,2])


# In[46]:


fig, axes = plt.subplots(1,2, figsize=(18, 15))

fig.suptitle('Distribution')

sns.boxplot(ax=axes[ 0], data=topsubscribed_df, x='range_subs', y='Video Views')
sns.boxplot(ax=axes[ 1], data=topsubscribed_df, x='range_subs', y='Started')
plt.show()


# In[ ]:




