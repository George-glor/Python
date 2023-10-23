#!/usr/bin/env python
# coding: utf-8

# # Pandas
# Read "10 minutes to Pandas": https://pandas.pydata.org/docs/user_guide/10min.html before solving the exercises.
# We will use the data set "cars_data" in the exercises below. 

# In[ ]:


# Importing Pandas. 
import pandas as pd


# ### Explain what a CSV file is.

# In[ ]:


it's a file format allow us to write from python code to an exel file 


# ### Load the data set "cars_data" through Pandas. 

# In[ ]:


# When reading in the data, either you have the data file in the same folder as your python script
# or in a seperate folder.

# Code below can be ran if you have the data file in the same folder as the script
# cars = pd.read_csv("cars_data.csv")

# Code below can be ran if you have the data file in another script. 
# Notice, you must change the path according to where you have the data in your computer. 
# pd.read_csv(r'C:\Users\Antonio Prgomet\Documents\ec_utbildning\kursframstallning\ds23\python_stat\exercises\numpy_matplot_pandas\cars_data.csv')


# ### Print the first 10 rows of the data. 

# In[ ]:


cars = pd.read_csv("cars_data.csv")
print(cars.head(10))


# ### Print the last 5 rows. 

# In[ ]:


print(cars.tail(5))


# ### By using the info method, check how many non-null rows each column have. 

# In[ ]:


cars.info()


# ### If any column has a missing value, drop the entire row. Notice, the operation should be inplace meaning you change the dataframe itself.

# In[ ]:


cars.dropna(inplace=True)


# ### Calculate the mean of each numeric column. 

# In[ ]:


means = cars.mean()
print(means)


# ### Select the rows where the column "company" is equal to 'honda'. 

# In[ ]:


honda_cars = cars[cars['company'] == 'honda']
print(honda_cars)


# ### Sort the data set by price in descending order. This should *not* be an inplace operation. 

# In[ ]:


sorted_cars = cars.sort_values(by='price', ascending=False)
print(sorted_cars)


# ### Select the rows where the column "company" is equal to any of the values (audi, bmw, porsche).

# In[ ]:


selected_cars = cars[cars['company'].isin(['audi', 'bmw', 'porsche'])]
print(selected_cars)


# ### Find the number of cars (rows) for each company. 

# In[ ]:


company_counts = cars.groupby('company').size()
print(company_counts)


# ### Find the maximum price for each company. 

# In[ ]:


company_max_prices = cars.groupby('company')['price'].max()
print(company_max_prices)

