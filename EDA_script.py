# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:46:03 2025

@author: Anjali Gangotri
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#read the csv file
df=pd.read_csv(r"C:\Users\ banking_data.csv")
# print(df)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

print(df.info())
print(df.isnull().sum())
print(df.columns)#marital and marital status seem like they contain the same data
print(df['marital'].value_counts())
print(df['marital_status'].value_counts())
#both the above columns have same values therefore one of is redundant and can be dropped
df.drop(columns=['marital'],inplace=True)
print(df.columns)
df['marital_status']=df['marital_status'].fillna(df['marital_status'].mode()[0])#since marital_status is a categorical variable, we impute it with the most frequent mode at 0 index
print(df['marital_status'].isnull().sum())#we check the number of missing values and we find that it is now 0 for marital_status
print(df['education'].value_counts())
print(df['education'].isnull().sum())#To check for null values
#Let's impute the nan values with the mode since education is a categorical variable
df['education']=df['education'].fillna(df['education'].mode()[0])
print(df['education'].isnull().sum())#Now the missing values are zero
#Let's delete the unknown values
df.drop(df[df['education'] == 'unknown'].index, inplace=True)
print(df['education'].value_counts())#Now the missing rows have been successfully deleted
#Let's rename some columns for easier understanding
df.rename(columns={'loan':'personal_loans'},inplace=True)
df.rename(columns={'housing':'housing_loans'},inplace=True)
df.rename(columns={'campaign':'number_of_contacts'},inplace=True)
df.rename(columns={'previous':'pcontacts'},inplace=True)
df.rename(columns={'y':'subscribed'},inplace=True)
print(df.columns)


#1. Distribution of age
print(df['age'].isnull().sum())#To check the number of null values
min_age=df['age'].min()
max_age=df['age'].max()
print(f"The range of age is {min_age} to {max_age}")
print(df['age'].describe())
plt.hist(x=df['age'],bins=10,color='purple',edgecolor='black')
plt.xlabel("Age of Bank Client")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.show()


#2. How does the job vary among the clients
print(df['job'].isnull().sum())#To check if any null values are present that needed to be deleted or imputed
print("The different jobs are:")
print(df['job'].unique())
print(df['job'].describe())
sns.countplot(x='job',data=df, edgecolor='black' )
plt.xticks(x=df['job'], rotation =45)
plt.xlabel("Job of Bank Client")
plt.ylabel("Frequency")
plt.title("Distribution of Jobs")
plt.figure(figsize=(10,6))
plt.show()


#It's interesting to note how the jobs vary according to level of education of the clients
print(df.groupby(['job','education'])['education'].count())
sns.catplot(x='job',hue='education',data=df,kind='count',palette=['blue','yellow','red'])
plt.xlabel("Jobs")
plt.ylabel('Frequency')
plt.title('Distribution of Jobs by level of Education')
plt.xticks(rotation=45)
plt.figure(figsize=(10,8))
plt.show()


#Clients in management often require mastersdegree which is justified by their highest count in tertiary education
#Wheresas clients with blue collar jobs and housemaids mostly have primary to secondary education only
#It's also interesting to see how personal loans vary with jobs of clients
sns.catplot(x='job',hue='personal_loans',data=df,kind='count',palette=['red','green'])
plt.xlabel("Job")
plt.ylabel('Frequency')
plt.title('Distribution of Jobs by personal loans')
plt.xticks(rotation=45)
plt.figure(figsize=(10,8))
plt.show()
print(df.groupby(['job','personal_loans'])['job'].count())


#admin=24,blu-21, ent-31, hou-14, man-15, ret-16,self- 17,serv-25, stud-1.3, tech-21, unk-2.5
#It turns out that proportionally, entrepreneurs or clients in service-sectors take highest amounts of personal loans. This makes sense since they need funding to start their business
#It's also interesting to see which clients have the most housing loans according to the jobs
sns.catplot(x='job',hue='housing_loans',data=df,kind='count',palette=['green','red'])
plt.xlabel("Jobs")
plt.ylabel('Frequency')
plt.title('Distibution of Jobs by housing loans')
plt.xticks(rotation=45)
plt.figure(figsize=(10,8))
plt.show()
#It turns out that people with blue-collar jobs have the most housing loans since they have a lesser pay.
#Students have least housing loans
#It's also notable that clients who have credit in default don't tend to have/get personal loans.
print(df.groupby(['job','default','personal_loans'])['personal_loans'].count())
sns.catplot(x='job',hue='personal_loans',col='default',data=df,kind='count',palette=['red','green'])
plt.xlabel("Jobs")
plt.ylabel('Frequency')
plt.title('Distribution of Jobs by personal loans')
plt.suptitle('Default')
plt.xticks(rotation=45,)
plt.figure(figsize=(20,15))
plt.show()


#3 marital status distribution of the clients
#Lets deal with missing values
print(df['marital_status'].isnull().sum())#To check is any null values are present
df['marital_status'].describe()
#Lets plot the distribution of marital status
sns.countplot(x='marital_status',data=df, edgecolor='black')
plt.xlabel('Marital Status of the Clients')
plt.ylabel('Frequency')
plt.title('Marital Status Distribution')
plt.show()
#Let's plot marital status by housing loans.
sns.catplot(x='marital_status',hue='housing_loans',data=df,palette=['red','green'][::-1],kind='count')
plt.xlabel("Marital Status")
plt.ylabel('Frequency')
plt.title('Distribution of Marital Status by Housing Loans')
plt.figure(figsize=(10,8))
plt.show()
#It's interesting to note how married clients tend to have a higher number of housing loans as compared to single or divorced clients since they are starting a family and would likely take up a loan for a house


#4 level of education among the clients
sns.histplot(x='education', data=df, color='orange',edgecolor='black')
plt.xlabel('Level of Education')
plt.ylabel('Frequency')
plt.title('Distribution of Education Level among Bank Clients')
plt.show()


#5 Credit in default
print(df['default'].value_counts())
print("The number of clients that have credit in default are:",df[df['default']=='yes'].shape[0],"the clients wihtout credit in default are:",df[df['default']=='no'].shape[0])
sns.countplot(data=df,x='default',edgecolor='black',palette=['green','red'][::-1])
plt.xlabel('Clients with credit in default')
plt.ylabel('Frequency')
plt.title('Distribution of Clients with credit in default')
plt.show()


#6 Distribution of Average yearly balance
print(df['balance'].nunique())
print(df['balance'].dtype)
print(df['balance'].isnull().sum())
print(df['balance'].describe())
plt.hist(x=df['balance'],edgecolor='black',color='yellow')
plt.xlabel('Average yearly balance')
plt.ylabel('Frequency')
plt.title('Distribution of Balance among Clients')
plt.figure(figsize=(10,8))
plt.show()


#7 How many clients have housing loans?
print(df['housing_loans'].value_counts())
print("Number of clients having housing loans are:",df[df['housing_loans']=='yes'].shape[0])


#8 How many clients have personal loans?
print(df.columns)
df.rename(columns={'loan':'personal_loans'},inplace=True)
print(df.columns)
print(df['personal_loans'].value_counts())
print("Number of clients having Personal loans are:",df[df['personal_loans']=='yes'].shape[0])


#9 What are the communication types used for contacting clients during the campaign?
print("The different communicatin types are",df['contact'].unique())
sns.countplot(data=df,x='contact',edgecolor='black')
plt.xlabel("Different Communicaton types")
plt.ylabel('Frequency')
plt.title("Distribution of Different Communication types")
plt.figure(figsize=(10,8))
plt.show()


#10 What is the distribution of the last contact day of the month?
print(df['day'].nunique(),df['day'].dtype)#contains each day of the month even weekends
sns.histplot(x='day',bins=4,data=df)
plt.xlabel("Days of the Month")
plt.ylabel('Frequency')
plt.title(" Weekwise Distribution of the Last Contact Day of the month")
plt.figure(figsize=(10,8))
plt.show()


#11 How does the last contact month vary among clients?
print(df['month'].nunique(),df['month'].dtype)
#It contains all 12 months as the last contact month
sns.countplot(x='month',data=df,edgecolor='black')
plt.xlabel("Months")
plt.ylabel('Frequency')
plt.title('Distibution of Last Contact Months')
plt.xticks(rotation=45)
plt.figure(figsize=(10,8))
plt.show()


#12 Distribution of duration of last contact
print(df['duration'].nunique(),df['duration'].dtype)
print(df['duration'].describe())
plt.hist(x=df['duration'],bins=20, edgecolor='black',color='green')
plt.xlabel('Duration of Last Contact')
plt.ylabel('Frequency')
plt.title('Distribution of Duration of Last Contact')
plt.show()


#13 How many contacts were performed during the campaign for each client?
df.rename(columns={'campaign':'number_of_contacts'},inplace=True)#delete this
print(df['number_of_contacts'].nunique())
print(df['number_of_contacts'].describe())
sns.histplot(x='number_of_contacts',bins=10,data=df, edgecolor='black',color='orange')
plt.xlabel("Number of Contacts in the Campaign")
plt.ylabel('Frequency')
plt.title('Distibution of Number of Contacts')
plt.figure(figsize=(10,8))
plt.show()


#14 What is the distribution of the number of days passed since the client was last contacted from a previous campaign?
print(df['pdays'].nunique())
print(df['pdays'].describe())
plt.hist(x=df['pdays'],bins=20,edgecolor='black',color='skyblue')
plt.xlabel('Days passed since Last Contact')
plt.ylabel('Frequency')
plt.title('Distribution of Days passed since Last Contact')
plt.show()
sns.boxplot(y='pdays',data=df)
plt.title('Box Plot for days passed since previous contact')
plt.show()
#We know that a lot of outliers exist for this data
#The histogram doesn't give us a normally distributed graph and a good representation since the count of -1 or non-contacted clients shadows the other data
print(df['pdays'].sort_values().unique()[1])#This statement gives us the second lowest value in the pdays column
#We can use this as the range for distribution of pdays other than -1 to give a more accurate understanding of the distribution of the days passed.
plt.hist(x=df['pdays'],bins=20,range=[1,871],edgecolor='black',color='skyblue')
plt.xlabel('Days passed since Last Contact')
plt.ylabel('Frequency')
plt.title('Distribution of Days passed since Last Contact')
plt.show()
print("The number of clients not previously contacted are,",df[df['pdays']==-1].shape[0])


#This gives us a better represntation
#15 How many contacts were performed before the current campaign for each client?
df.rename(columns={'previous':'pcontacts'},inplace=True)
print(df.columns)
print(df['pcontacts'].nunique())
print(df['pcontacts'].describe())
sns.histplot(x='pcontacts',bins=20,data=df, edgecolor='black',color='violet')
plt.xlabel("Number of Previous Contacts")
plt.ylabel('Frequency')
plt.title('Distibution of Number of Previous Contacts')
plt.figure(figsize=(10,8))
plt.show()


#16 What were the outcomes of the previous marketing campaigns?
print("The different outcomes of the previous campaigns are,",df['poutcome'].unique())
sns.countplot(x='poutcome',data=df, edgecolor='black')
plt.xlabel('Outcome of Previous Campaigns')
plt.ylabel('Frequency')
plt.title('Distribution of outcomes of Previous Campaigns')
plt.figure(figsize=(10,8))
plt.show()
#17 What is the distribution of clients who subscribed to a term deposit vs. those who did not?
df.rename(columns={'y':'subscribed'},inplace=True)
print(df['subscribed'].value_counts())
sns.countplot(x='subscribed',data=df,palette=['red','green'])
plt.xlabel('Clients subscribed to Term Deposit')
plt.ylabel('Frequency')
plt.title('Distribution of Clients who Subscribed to Term Deposit')
plt.figure(figsize=(10,8))
plt.show()


#18 Correlation Matrix of Different Atrributes to Subscribed to Term Depositprint(df.select_dtypes(include=['int64','float64']).head(10))
df['subscribed'].replace({'yes':1,'no':0},inplace=True)
df['personal_loans'].replace({'yes':1,'no':0},inplace=True)
df['housing_loans'].replace({'yes':1,'no':0},inplace=True)
print(df.select_dtypes(include=['int64','float64']).columns)
numeric_df=df.select_dtypes(include=['int64','float64'])
corr_matrix=numeric_df.corr()
plt.figure(figsize=(10,8))
plt.title('Correlation matrix of Banking Clients Dataset')
sns.heatmap(data=corr_matrix,annot=True,cmap='PuBuGn',fmt='.2f')
plt.show()


#The correlation matrix gives us the below useful insights:
#A moderate positive correlation shows that higher is the duration of the call, more are the chances of subscription by the client. Thus longer calls tend to be more effective.
#A weak correlation exists but it is positive nonetheless and shows that higher is the number of contact in previous campaigns, greater are the chances of succssful subscription to term deposit. This might be because the client forms a relationship with you and gains confidence that the company will deliver good results
#A weak negative correlation shows that clients with a housing or a personal loan don't tend to subscribe to a term deposit.
#If more days have passed since previous contact, the term of the client is likely to have ended in the interval and the client might be looking for another term deposit
#A weak negative correlation shows that a lot of phone calls or contacts with the same client in the same campaign is futile.
#Insights not Related to the Target Variable
#more is the age lesser are the housing loans
#more are the loans taken- housing or personal, lesser is the balance remaining.
print(df['job'].unique())
sns.countplot(x='job',data=df,hue='subscribed')
plt.xticks(rotation=45)
plt.show()


#Clients in management,students or retired clients tend to subscribe to term deposits.
#When we look at the approximate proportion using groupby method, we find the following proportions: management: 0.34,technician: 0.1,entrepreneur: 0.05,retired: 0.27,admin: 0.13,services: 0.08,blue-collar: 0.083,self-employed: 0.09,housemaid- 0.14...
sns.countplot(x='education',data=df,hue='subscribed')
plt.xticks(rotation=45)
plt.show()
#By proportion, Tertiary educated people are more aware and tend to subscribe to a term deposit.
print(df.groupby(['marital_status','subscribed'])['subscribed'].count())
sns.countplot(x='marital_status',data=df,hue='subscribed')
plt.show()
#Married couples should ideally subscribe more to generate more money for their family but when we look at the approximate proportion using groupby method, we find the following proportions: married: 11.1%, single: 17%, divorced:13%
#Thus single clients tend to subscribe more.
print(df.groupby(['contact','subscribed'])['subscribed'].count())
sns.countplot(x='contact',data=df,hue='subscribed')
plt.show()
#Using groupby, we see the following proportions: cellular: 17%, telephone: 14%. Thus people contacted on cell phones tend to subscribe more compared to people contacted on telephones
sns.countplot(x='month',data=df,hue='subscribed')
plt.show()
sns.countplot(x='poutcome',data=df,hue='subscribed')
plt.show()
#success-more ppl subscribe, failiure- less people subscribe
sns.histplot(data=df,x='age',hue='subscribed',bins=2)
plt.title('Variation in age by subscription')
plt.show()
#Thus by proportion, old clients above the age of 57 tend to subscribe more than younger clients