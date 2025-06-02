#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


import pandas as pd
a = pd.read_csv(r"C:\Users\rm136\Downloads\sales.csv", encoding='latin1')


# In[3]:


a.head(5)


# In[4]:


a=a.drop('ORDERNUMBER',axis=1)


# In[5]:


a=a.drop('ORDERDATE',axis=1)


# In[6]:


a=a.drop('ADDRESSLINE1',axis=1)


# In[7]:


a=a.drop('ADDRESSLINE2',axis=1)


# In[8]:


a=a.drop('POSTALCODE',axis=1)


# In[9]:


a=a.drop('CONTACTLASTNAME',axis=1)


# In[10]:


a=a.drop('CONTACTFIRSTNAME',axis=1)


# In[11]:


a=a.drop('PHONE',axis=1)


# In[12]:


a=a.drop('PRODUCTCODE',axis=1)
a=a.drop('CUSTOMERNAME',axis=1)


# In[13]:


a.tail(3)


# In[14]:


a.info()


# In[15]:


a.describe()


# In[16]:


# Group by PRODUCTLINE and count the number of unique MSRP values
unique_msrp_counts = a.groupby('PRODUCTLINE')['MSRP'].nunique().reset_index()

# Rename the column for clarity
unique_msrp_counts.columns = ['PRODUCTLINE', 'Unique_MSRP_Count']

# Display the result
print(unique_msrp_counts)


# In[17]:


# Group by PRODUCTLINE and calculate average MSRP
avg_msrp = a.groupby('PRODUCTLINE')['MSRP'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(avg_msrp['PRODUCTLINE'], avg_msrp['MSRP'], color='skyblue')
plt.xlabel('PRODUCTLINE')
plt.ylabel('Average MSRP')
plt.title('Average MSRP by PRODUCTLINE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[18]:


statecnt=a['STATE'].value_counts().reset_index()
print(statecnt)


# In[19]:


nullstate=a['STATE'].isna().sum()
print(nullstate)
#if i want to replace this with using mode but their is an problem because the Null values amount is
#too big like half of columnn entries are NaN its is risky the if state is not matter much in your
# dataset so left this column and analyse it with another column like city, country etc.
a=a.drop('STATE',axis=1)


# In[20]:


country_count=a['COUNTRY'].value_counts().reset_index()
print(country_count)


# In[21]:


citycnt=a['CITY'].value_counts().reset_index()
print(citycnt)
nacity=a['CITY'].isna().sum()
print(nacity)


# In[22]:


cntstatus=a['STATUS'].value_counts().reset_index()
print(cntstatus)
nastatus=a['STATUS'].isna().sum()


# In[23]:


statuscnt=a['STATUS'].value_counts().reset_index()
statuscnt.columns=['STATUS','count']
plt.figure(figsize=(10,6))
plt.bar(statuscnt['STATUS'],statuscnt['count'],color='skyblue')
plt.xticks(rotation=45)#for rotate status name to seen good don't overlap
plt.tight_layout()#automatically adjusts spacing so nothing gets cut off.
plt.show()


# In[24]:


a.head(2)


# In[25]:


yearct=a['YEAR_ID'].value_counts().reset_index()
print(yearct)


# In[26]:


sales_sum = a.groupby('PRODUCTLINE')['SALES'].sum()
plt.figure(figsize=(4,4))

plt.pie(sales_sum, labels=sales_sum.index,radius=0.2,autopct='%1.1f%%')
plt.title('Sales Distribution by Product Line')
plt.axis('equal')  # Make it a perfect circle
plt.show()


# In[27]:


quat_odr=a.groupby('PRODUCTLINE')['QUANTITYORDERED'].sum()
plt.figure(figsize=(4,4))
plt.pie(quat_odr,labels=quat_odr.index,radius=0.7)
plt.show()


# In[28]:


city=a['CITY'].value_counts().reset_index()
print(city)


# In[29]:


mdrid_data=a[a['CITY']=='Madrid']
mdrid_sales_sum=mdrid_data.groupby('PRODUCTLINE')['SALES'].sum()
plt.bar(mdrid_sales_sum.index,mdrid_sales_sum.values,color='k')
plt.xticks(rotation=45)
plt.tight_layout()


# In[30]:


#in this i want to analysis the 2004 with the basis of month and sales
sales2004=a[a['YEAR_ID']==2004]
sales_bymnt=sales2004.groupby('MONTH_ID')['SALES'].sum()
#so the month is shown by is the number firsly we want to convert it in the name

month_nm={
    1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
sales_bymnt.index=sales_bymnt.index.map(month_nm)

mnt_odr=['January','February','March','April','May','June','July','August','September','October','November','December']
sales_bymnt=sales_bymnt.reindex(mnt_odr)

plt.figure(figsize=(10,6))
plt.plot(sales_bymnt.index,sales_bymnt.values,marker='o',linestyle='-',color='green')
plt.xlabel('2004 Months')
plt.ylabel('2004 Sales according months')
plt.title('Analysing the 2004 sales sepretely')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[31]:


#in this we analysis the 2005 for we thought that this the loss year 
sales2005=a[a['YEAR_ID']==2005]
sales_bymnt=sales2005.groupby('MONTH_ID')['SALES'].sum()

month_nm={
    1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
sales_bymnt.index=sales_bymnt.index.map(month_nm)

mnt_odr=['January','February','March','April','May','June','July','August','September','October','November','December']
sales_bymnt=sales_bymnt.reindex(mnt_odr)

plt.figure(figsize=(10,6))
plt.plot(sales_bymnt.index,sales_bymnt.values,color='red',linestyle='-',marker='*')
plt.xlabel('2005 Months')
plt.ylabel('2005 Sales according months')
plt.title('Analysing the 2005 sales sepretely by months')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



# We Analyse that there are only 5 months in 2005 so we create a double line plot and analyse the different between 2004 and 2005 sales on the basis of our 5 months.

# In[32]:


data_4=a[(a['YEAR_ID']==2004)&(a['MONTH_ID']<=5)]
data_5=a[(a['YEAR_ID']==2005)&(a['MONTH_ID']<=5)]

sales4=data_4.groupby('MONTH_ID')['SALES'].sum()
sales5=data_5.groupby('MONTH_ID')['SALES'].sum()


plt.figure(figsize=(10,6))
plt.plot(sales4.index,sales4.values,marker='o',linestyle='-',color='green',label='2004 sales(lakhs)')
plt.plot(sales5.index,sales5.values,marker='*',linestyle='-',color='red',label='2005 sales(lakhs)')
plt.xlabel('MONTH')
plt.ylabel('SALES IN LAKHS')
plt.title('Comparison between 2004 and 2005 sales')
plt.legend()
plt.tight_layout()
plt.xticks([1,2,3,4,5],['January','February','March','April','May'])
plt.grid(True)
plt.show()


# It shows that the 2005 is great sales of the begining months as compare to the 2004 but we have only 5 months of data in 2005.

# In[33]:


a.tail(2)


# In[34]:


trtcount=a['TERRITORY'].value_counts().reset_index()
print(trtcount)
a['TERRITORY'].isna().sum()


# In[35]:


qtrcount=a['QTR_ID'].value_counts().reset_index()
print(qtrcount)
#So this will show that higher sales is in Q4 like October, November, December.


# In[36]:


dealcnt=a['DEALSIZE'].value_counts().reset_index()
print(dealcnt)


# In[37]:


large_data=a[a['DEALSIZE']=='Large']
large_sales=large_data.groupby('PRODUCTLINE')['SALES'].sum()
plt.figure(figsize=(10,6))
plt.bar(large_sales.index,large_sales.values,color='g')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#IT shows the bar graph between the Product and sales in large deal size


# In[38]:


contorderlinenum=a['ORDERLINENUMBER'].value_counts().reset_index()
print(contorderlinenum)


# In[ ]:




