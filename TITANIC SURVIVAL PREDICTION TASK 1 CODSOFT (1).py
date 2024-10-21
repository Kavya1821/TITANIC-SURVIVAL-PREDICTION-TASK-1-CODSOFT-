#!/usr/bin/env python
# coding: utf-8

# In[34]:


#TASK 1 TITANIC SURVIVAL PREDICTION 
#CODSOFT 
#DOMAIN - DATA SCIENCE 
#BATCH - 25TH SEPTEMBER 2024- 25TH OCTOBER 2024 


# In[35]:


#IMPORTING IMPORTANT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[36]:


#IMPORTING DATASET 
df= pd.read_csv("Titanic-Dataset.csv")
df.head(10)


# In[37]:


df.info()


# In[38]:


df_num = df[["Age", "SibSp", "Parch", "Fare"]]
df_cat = df[["Survived", "Sex", "Cabin", "Embarked", "Ticket"]]


# In[39]:


for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[40]:


sns.barplot(data=df, x="Pclass", y="Fare", hue="Survived")


# In[41]:


pd.pivot_table(df, index="Survived", values=["Age", "SibSp", "Parch", "Fare"])


# In[42]:


for i in df_cat.columns:
    sns.barplot(x=df_cat[i].value_counts().index, y=df_cat[i].value_counts())
    plt.show()


# In[43]:


x = pd.DataFrame(
    (
        pd.pivot_table(
            df,
            index="Survived",
            columns="Sex",
            values="Ticket",
            aggfunc="count",
        )
    )
)
print()
print(
    pd.pivot_table(
        df, index="Survived", columns="Pclass", values="Ticket", aggfunc="count"
    )
)
print()
print(
    pd.pivot_table(
        df,
        index="Survived",
        columns="Embarked",
        values="Ticket",
        aggfunc="count",
    )
)
print()
x


# In[44]:


df.shape


# In[45]:


df.describe()


# In[46]:


#from the baove cell it is clear that there are few missing values in age column 


# In[47]:


df['Survived'].value_counts()


# In[48]:


#lets visualise the count of survival wrt pclass
sns.countplot(x=df['Survived'] , hue=df['Pclass'])


# In[49]:


df['Sex']


# In[50]:


#lets visualise the count of survivals wrt Gender 
sns.countplot(x=df['Sex'],hue = df['Survived'])


# In[51]:


#LOOK AT THE SURVIVAL RATE BY SEX
df.groupby('Sex')[['Survived']].mean()


# In[52]:


df['Sex'].unique()


# In[53]:


from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()

df['Sex'] = labelencoder.fit_transform(df['Sex'])

df.head()


# In[54]:


df['Sex'],df['Survived']


# In[55]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[56]:


#DATA CLEANING
df.isna().sum()


# In[57]:


#AFTER DROPPING NON REQUIRED COLUMN 
df=df.drop(['Age'],axis = 1)


# In[58]:


df_final = df 
df_final.head(10)


# In[59]:


#FEATURE ENGINNERING


# In[60]:


df["Fare"] = np.log(df["Fare"] + 1)
sns.displot(df["Fare"], kde=True)


# In[61]:


corr =df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")


# In[62]:


from sklearn.preprocessing import LabelEncoder

cols = ["Sex", "Embarked"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

df.head()


# In[63]:


X = df.drop(columns=["Survived"], axis=1)
y = df["Survived"]
df


# In[64]:


#MODEL TRAINING 


# In[ ]:





# In[65]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load a dataset (e.g., the iris dataset)
df = load_iris()
X = df.data  # Features
y = df.target  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier
model = DecisionTreeClassifier()

# Train the model (fit it to the data)
model.fit(X_train, y_train)

# Use the model to make predictions on the test set
predictions = model.predict(X_test)

# Output the predictions
print(predictions)


# In[66]:


print(y_test)


# In[67]:


pred = model.predict(X_test) 
print(pred)


# In[68]:


from sklearn.model_selection import train_test_split, cross_val_score


def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=40
    )
    model.fit(x_train, y_train)
    print("Accuracy", model.score(x_test, y_test))

    score = cross_val_score(model, X, y, cv=5)
    print("CV SCORE :", np.mean(score))


# In[69]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
classify(model)


# In[70]:


get_ipython().system('pip install lightgbm')
from lightgbm import LGBMClassifier

model = LGBMClassifier()
classify(model)


# In[71]:


get_ipython().system('pip install xgboost')

from xgboost import XGBClassifier

model = XGBClassifier()
classify(model)


# In[72]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
classify(model)


# In[73]:


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
classify(model)


# In[74]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
classify(model)


# In[75]:


model = XGBClassifier()
model.fit(X, y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




