#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def wrangle(file):
    #Read csv file
    df = pd.read_csv(file)
    
    # Fill missing values in the 'Age' column with the median age
    df["Age"] = df["Age"].fillna(df["Age"].median()).astype(int)
    
    # Fill missing values in the 'Cabin' column with the most frequent value
    df["Cabin"] = df["Cabin"].fillna(df["Cabin"].mode()[2])
    
    #Fill missing values in the 'Embarked' column with the most common embarkation point
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    #Set the passengerID to index
    df= df.set_index(df["PassengerId"]).sort_index(ascending=True)
    df.drop(columns="PassengerId", inplace=True)

    return df


# In[3]:


df = wrangle("train.csv")
print(df.shape)
df.head()


# In[4]:


# Get an overview of the data
print(df.info())
df.head()


# In[5]:


df.describe()


# In[5]:


#Total Number of Passengers
print("Total Number of Passenger:",len(df))
print("Total Number of Male Passeneger:",df[df["Sex"]=="male"].value_counts().sum())
print("Total Number of Female Passeneger:", df[df["Sex"]=="female"].value_counts().sum())


# In[6]:


sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Sex')
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()


# In[7]:


#Create a datframe for the survived passengers
survived_passengers = df[df['Survived'] == 1][["Sex","Age", "Pclass", "Embarked"]]
survived_passengers = pd.DataFrame(survived_passengers)
survived_passengers.head(10)


# In[21]:


print(f"Number of Passengers who survived: {df.Survived.value_counts()[1]}")
print(f"Number of Passengers who didn't survive: {df.Survived.value_counts()[0]}")
sex = survived_passengers["Sex"].value_counts().to_frame()
sex.plot(kind="bar")
plt.title('Survival by Gender')
plt.xlabel("Sex")
plt.ylabel("Count");
survived_passengers["Sex"].value_counts().to_frame()


# In[51]:


print("Total Number of Passengers in the First Class:",df[df["Pclass"] == 1].value_counts().sum(), "Passengers")
print("Total Number of Passengers in the Second Class:",df[df["Pclass"] == 2].value_counts().sum(), "Passengers")
print("Total Number of Passengers in the Third Class:",df[df["Pclass"] == 3].value_counts().sum(), "Passengers")


# In[23]:


sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Class')
plt.xlabel("Survival")
plt.ylabel("Count");
plt.show()


# In[32]:


pclass1 = (survived_passengers[survived_passengers["Pclass"] == 1]["Sex"].value_counts().sum())
print(f"Total Number of Survivors in First Class: {pclass1} Passengers")
pclass2 = (survived_passengers[survived_passengers["Pclass"] == 2]["Sex"].value_counts().sum())
print(f"Total Number of Survivors in Second Class: {pclass2} Passengers")
pclass3 = (survived_passengers[survived_passengers["Pclass"] == 3]["Sex"].value_counts().sum())
print(f"Total Number of Survivors in Thrid Class: {pclass3} Passengers")

pclass = survived_passengers["Pclass"].value_counts().to_frame().sort_index()
pclass.plot(kind="bar")
plt.title('Survived Pclass')
plt.xlabel("P-Class")
plt.ylabel("Count");


# In[311]:


sns.countplot(x='Sex', hue='Pclass', data=survived_passengers)
plt.title('Gender in the Pclass that Survived')
plt.ylabel("Frequency")
plt.show()

survived_passengers[["Pclass", "Sex"]].value_counts().to_frame().sort_index()


# In[38]:


sns.countplot(x='Survived', hue='Embarked', data=df)
plt.title('Survival by Embarkment')
plt.xlabel("Survival")
plt.ylabel("Count");


# In[63]:


embark = survived_passengers["Embarked"].value_counts().to_frame()
embark.plot(kind="barh")
plt.title('Total Number of Survivor per Embarkment')
plt.xlabel("Frequency")
plt.ylabel("Embark Port")
survived_passengers["Embarked"].value_counts().to_frame().sort_values(by="count")


# In[70]:


survived_passengers[["Embarked", "Pclass", "Sex"]].value_counts().to_frame().sort_index()


# In[41]:


sns.scatterplot(data=df, x="Age", y="Fare", hue="Age")
plt.title("Scatter plot of Age and Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived");


# In[59]:


survived_passengers["Age"].value_counts().sort_index().to_frame().plot()
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Count");


# In[ ]:




