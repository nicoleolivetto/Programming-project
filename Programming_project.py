import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from PIL import Image
import io
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection  import train_test_split

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

def get_downloadable_data(df):
    return df.to_csv().encode('utf-8')

st.header('Top 1000 Most Subscribed Youtube Channels')

topsubscribed_df =pd.read_csv('/Users/nicoleolivetto/Downloads/topSubscribed.csv')

st.write(topsubscribed_df)

st.download_button('download raw data', get_downloadable_data(topsubscribed_df), file_name='Top_subscribed.csv')

show_info=st.checkbox('Show info()')
if show_info:
    st.subheader('Topsubscribed_df.info()')


    buffer = io.StringIO()
    topsubscribed_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

topsubscribed_df.rename({'Video Views': 'VideoViews'}, axis=1, inplace=True)

topsubscribed_df['Subscribers']=topsubscribed_df['Subscribers'].str.replace(',','')
topsubscribed_df['VideoViews']=topsubscribed_df['VideoViews'].str.replace(',','')
topsubscribed_df['Video Count']=topsubscribed_df['Video Count'].str.replace(',','')

topsubscribed_df['Subscribers'] = topsubscribed_df['Subscribers'].astype('int64')
topsubscribed_df['VideoViews'] = topsubscribed_df['VideoViews'].astype('int64')
topsubscribed_df['Video Count'] = topsubscribed_df['Video Count'].astype('int64')

topsubscribed_df=topsubscribed_df[(topsubscribed_df != 0).all(1)]


topsubscribed_df = topsubscribed_df.drop(topsubscribed_df[topsubscribed_df['Started'] < 2005].index)

topsubscribed_df = topsubscribed_df.replace('https://us.youtubers.me/global/all/top-1000-most_subscribed-youtube-channels','Unknown')

topsubscribed_df.reset_index(drop=True, inplace=True)

st.subheader('Dataset after chanching column types, removing rows with zeros,changing the name of a column, removing raw with wrong value')

st.write(topsubscribed_df)

show_infos=st.checkbox('Show info() of new df')
if show_infos:
    st.subheader('Topsubscribed_df.info()')


    buff = io.StringIO()
    topsubscribed_df.info(buf=buff)
    n = buff.getvalue()
    st.text(n)


show_info=st.checkbox('Show describe()')
if show_info:
    st.subheader('Topsubscribed_df.describe()')
    st.write(topsubscribed_df.describe())

st.subheader('I created a new column name "range_subs" and divided all entries in 6 categories')

topsubscribed_df=topsubscribed_df.assign(range_subs=0)
topsubscribed_df.range_subs = topsubscribed_df.range_subs.astype(str)

#add code
code = '''for i in range ((len(topsubscribed_df["Subscribers"]))):
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
'''

show_code=st.sidebar.checkbox('Show code()')
if show_code:
    st.subheader('Topsubscribed_df.describe()')
    st.code(code, language='python')


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

st.write(topsubscribed_df)

st.subheader('The following graphs show how many channels there are in each subscriber range')


fig1=plt.figure(figsize=(15,10))
plt.title('Subscriber range')
sns.set_style('whitegrid')
m = sns.barplot(x=topsubscribed_df['range_subs'].value_counts().index, y=topsubscribed_df['range_subs'].value_counts(), palette ='Paired')
m.set_xticklabels(topsubscribed_df['range_subs'].value_counts().index, rotation = 45)
m.set(xlabel ='range_subs', ylabel = 'Values')
plt.show()
   
st.pyplot(fig1)
    
st.set_option('deprecation.showPyplotGlobalUse', False)
pie_chart = topsubscribed_df['range_subs'].value_counts()
pie_chart.plot.pie(shadow=True,startangle=90,autopct='%1.1f%%')
plt.legend(topsubscribed_df['range_subs'])
plt.show()

st.pyplot()
    

st.subheader('How many of these youtubers joined YT each year?')
 

col_1, col_2 = st.columns(2)
with col_1: 
    annual_join = topsubscribed_df.groupby(['Started']).size().reset_index().rename(columns = {0:'Counts'})
    st.write(annual_join)

with col_2:
    fig2=plt.figure(figsize=(15,10))
    plt.title('join per year')
    sns.set_style('whitegrid')
    a=sns.barplot(x=topsubscribed_df['Started'].value_counts().sort_index().index, y=topsubscribed_df['Started'].value_counts())
    a.set_xticklabels(topsubscribed_df['Started'].value_counts().index, rotation = 45)
    a.set(xlabel ='Started', ylabel = 'Values')
    plt.show()

    st.pyplot(fig2)


st.subheader('How many Youtube channels in each category?')

col_1, col_2 = st.columns(2)
with col_1: 
    a=topsubscribed_df.groupby(['Category']).size().reset_index().rename(columns = {0:'Counts'})
    st.write(a)

with col_2:
    fig3=plt.figure(figsize=(15,10))
    plt.title('Categories')
    sns.set_style('whitegrid')
    n = sns.barplot(x=topsubscribed_df['Category'].value_counts().sort_index().index, y=topsubscribed_df['Category'].value_counts())
    n.set_xticklabels(topsubscribed_df['Category'].value_counts().index, rotation = 90)
    n.set(xlabel ='Started', ylabel = 'Values')
    #plt.show()

    st.pyplot(fig3)



col_1, col_2 = st.columns(2)

with col_1:
    st.subheader('Total view per category')
    st.write(topsubscribed_df.groupby("Category")["VideoViews"].sum().sort_values(ascending=False))
with col_2:
    st.subheader('Mean views per category')
    st.write(topsubscribed_df.groupby("Category")["VideoViews"].mean().sort_values(ascending=False))


st.subheader('Mean video count per sub range')

#stampo la media di videocount per pgni sub range
st.write( topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Count'].transform('mean').unique())
fig4, ax = plt.subplots()
#rappresesnto  come cambia la media di video count a seconda del sub range
ax.plot(topsubscribed_df['range_subs'].unique(), topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Count'].transform('mean').unique())
ax.set_xlabel("range")
ax.set_ylabel("media video count")
#plt.show()
st.pyplot(fig4)

st.subheader('Mean video views per sub range')

#stampo la media di video views per pgni sub range
st.write( topsubscribed_df.groupby(topsubscribed_df['range_subs'])['VideoViews'].transform('mean').unique())
fig5, ax = plt.subplots()
#rappresesnto  come cambia la media di video views a seconda del sub range
ax.plot(topsubscribed_df['range_subs'].unique(), topsubscribed_df.groupby(topsubscribed_df['range_subs'])['VideoViews'].transform('mean').unique())
ax.set_xlabel("range")
ax.set_ylabel("media video views")
st.pyplot(fig5)

st.subheader('How do attributes influence each other?')
fig6=sns.set(style="ticks", color_codes=True)    
sns.pairplot(topsubscribed_df)
st.pyplot(fig6)

st.subheader('Correlation matrix')

fig7=plt.figure(figsize=(8,6))
sns.heatmap(topsubscribed_df.corr(), annot=True)
st.pyplot(fig7)

st.subheader('How does the video count change depending on the sub range?')
fig8=plt.figure(figsize=(25,15))
sns.lineplot(x = 'range_subs', y = 'Video Count', data=topsubscribed_df)
plt.title('Video count nei vari range di subs')
plt.grid()
st.pyplot(fig8)

st.subheader('How do video views change depending on the sub range?')
#come il video views cambia a seconda del range subs
fig9=plt.figure(figsize=(25,15))
sns.lineplot(x = 'range_subs', y = 'VideoViews', data=topsubscribed_df)
plt.title('come il video views cambia a seconda del range subs')
plt.grid()
st.pyplot(fig9)

fig10=plt.figure(figsize=(30,10))
plt.subplot(121)

#come il range subs influisce su video views e su video count
st.subheader('How does the sub range affect the videos views and video count?')
plt.title('Video views depending on range subs',fontsize = 20)
plt.ylabel("videoviews")
topsubscribed_df.groupby('range_subs')['VideoViews'].mean().sort_index().plot.bar(color = 'pink')
plt.grid()
plt.subplot(122)
plt.title('Video count depending on range subs',fontsize = 20)
plt.ylabel("video count")
topsubscribed_df.groupby('range_subs')['Video Count'].mean().sort_index().plot.bar(color = 'yellow')
plt.grid()
st.pyplot(fig10)

r1=(topsubscribed_df[topsubscribed_df['range_subs']=='10M - 15M'])
r2=(topsubscribed_df[topsubscribed_df['range_subs']=='15M - 20M'])
r3=(topsubscribed_df[topsubscribed_df['range_subs']=='20M - 50M'])
r4=(topsubscribed_df[topsubscribed_df['range_subs']=='50M - 100M'])
r5=(topsubscribed_df[topsubscribed_df['range_subs']=='100M - 200M'])
r6=(topsubscribed_df[topsubscribed_df['range_subs']=='200M+'])

st.subheader('Correlation matrix for each sub range')

fig11, axes = plt.subplots(2,3, figsize=(25, 15))

sns.heatmap(r1.corr(), annot=True,ax=axes[0,0])
sns.heatmap(r2.corr(), annot=True,ax=axes[0,1])
sns.heatmap( r3.corr(), annot=True,ax=axes[0,2])
sns.heatmap(r4.corr(), annot=True,ax=axes[1,0])
sns.heatmap( r5.corr(), annot=True,ax=axes[1,1])
sns.heatmap(r6.corr(), annot=True,ax=axes[1,2])
st.pyplot(fig11)


st.subheader('Distribution of video views and started for each sub range')
fig12, axes = plt.subplots(1,2, figsize=(18, 15))

fig12.suptitle('Distribution')

sns.boxplot(ax=axes[ 0], data=topsubscribed_df, x='range_subs', y='VideoViews', palette='pastel')
sns.boxplot(ax=axes[ 1], data=topsubscribed_df, x='range_subs', y='Started', palette='pastel')
st.pyplot(fig12)

st.subheader('Df ater adding a new column containing the number of yoears each channel has been on YT')
topsubscribed_df = topsubscribed_df.assign(Years_on_YT = 2023 - topsubscribed_df['Started'])

df2 = topsubscribed_df.drop('Category', axis=1)
df2 = df2.drop('Youtube Channel', axis=1)
df2 = df2.drop('range_subs', axis=1)
df2.reset_index(drop=True, inplace=True)

q = df2["Video Count"].quantile(0.75)
q_low = df2["Video Count"].quantile(0.25)
q_hi  = df2["Video Count"].quantile(0.75)

df_filtered = df2[(df2["Video Count"] < q_hi) & (df2["Video Count"] > q_low)]
df_filtered.reset_index(drop=True, inplace=True)

fig12.suptitle('Dataset after removing outliers')

st.write(df_filtered)
show_infos=st.checkbox('Show info() of df after removing outliers')
if show_infos:
    st.subheader('Topsubscribed_df.info()')


    buff = io.StringIO()
    df_filtered.info(buf=buff)
    n = buff.getvalue()
    st.text(n)

st.subheader('Correlation matrix of the new df')
fig13=plt.figure(figsize=(8,6))
sns.heatmap(df_filtered.corr(), annot=True)
st.pyplot(fig13)

topsubscribed_df = topsubscribed_df.assign(Years_on_YT = 2023 - topsubscribed_df['Started'])
topsubscribed_df = topsubscribed_df.assign(ThirtyM="0")

df2 = topsubscribed_df.drop('Category', axis=1)
df2 = df2.drop('Youtube Channel', axis=1)
df2 = df2.drop('range_subs', axis=1)
df2.reset_index(drop=True, inplace=True)

q = df2["Video Count"].quantile(0.75)
q_low = df2["Video Count"].quantile(0.25)
q_hi  = df2["Video Count"].quantile(0.75)

df_filtered = df2[(df2["Video Count"] < q_hi) & (df2["Video Count"] > q_low)]
df_filtered.reset_index(drop=True, inplace=True)

for i in range (len(df_filtered['ThirtyM'])):
    if((df_filtered.at[(i), 'Subscribers']) > 30000000):
        df_filtered.at[(i), 'ThirtyM'] = 1
    else:
        df_filtered.at[(i), 'ThirtyM'] = 0

si = df_filtered[df_filtered.ThirtyM == 1] #hanno piu di 30M subs
no = df_filtered[df_filtered.ThirtyM == 0] #hanno meno do 30M subs  

#MODEL

model = GaussianNB()
y=df_filtered.ThirtyM
choices = st.multiselect('Select features', ['VideoViews','Rank','Started'])
test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)
if len(choices) > 0 and st.button('RUN MODEL'):
    with st.spinner('Training...'):
        x = df_filtered[choices]
        y=y.astype('int')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=2)

        x_train = x_train.to_numpy().reshape(-1, len(choices))
        model.fit(x_train, y_train)

        x_test = x_test.to_numpy().reshape(-1, len(choices))
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.write(f'Accuracy = {accuracy:.2f}')

