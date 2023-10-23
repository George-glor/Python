#!/usr/bin/env python
# coding: utf-8

# # Matplotlib
# Read the tutorials: "Basic Usage" and "Pyplot Tutorial" in the link: https://matplotlib.org/stable/tutorials/index.html before solving the exercises below. The "Pyplot Tutorial" you do not read in detail but it is good to know about since the fact that there are two approaches to plotting can be confusing if you are not aware of both of the approaches.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# ### Plotting in Matplotlib can be done in either of two ways, which ones? Which way is the recommended approach?

# In[ ]:


MATLAB style interface
Object oriented interface


# ### Explain shortly what a figure, axes, axis and an artist is in Matplotlib.

# In[ ]:


Image: The image is a top layer containing all the objects in the story. It can have one or more axes and other artists. It represents the whole picture, and has multiple subplots if any.

Axes: Axes are a portion of the image used to plot the data. You can think of it as a story in a painting. The important point is that the image can have multiple axes, allowing for background painting or stacking.

Axes - Axes are two or three axis objects, which specify data boundaries and tick positions in the plot. They represent the X and Y (sometimes Z) axes on the same map.

Artists: Artists are the individual parts of the story, such as characters, text, snippets, graphics, etc. Used to create and modify the visual elements in the story


# ### When plotting in Matplotlib, what is the expected input data type?

# In[ ]:


When plotting in Matplotlib, the expected input data type for the most common plotting functions like plot,


# ### Create a plot of the function y = x^2 [from -4 to 4, hint use the np.linspace function] both in the object-oriented approach and the pyplot approach. Your plot should have a title and axis-labels.

# In[ ]:


x = np.linspace(-4, 4, 100)
y = x**2
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("y = x^2")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()


# ### Create a figure containing 2  subplots where the first is a scatter plot and the second is a bar plot. You have the data below. 

# In[ ]:


# Data for scatter plot
np.random.seed(15)
random_data_x = np.random.randn(1000)
random_data_y = np.random.randn(1000)
x = np.linspace(-2, 2, 100)
y = x**2

# Data for bar plot
fruit_data = {'grapes': 22, 'apple': 8, 'orange': 15, 'lemon': 20, 'lime': 25}
names = list(fruit_data.keys())
values = list(fruit_data.values())


# In[ ]:


np.random.seed(15)
random_data_x = np.random.randn(1000)
random_data_y = np.random.randn(1000)
fruit_data = {'grapes': 22, 'apple': 8, 'orange': 15, 'lemon': 20, 'lime': 25}
names = list(fruit_data.keys())
values = list(fruit_data.values())
fig, (scatter_ax, bar_ax) = plt.subplots(1, 2, figsize=(12, 4))
scatter_ax.scatter(random_data_x, random_data_y, alpha=0.5, color='blue')
scatter_ax.set_title('Scatter Plot')
scatter_ax.set_xlabel('X-axis')
scatter_ax.set_ylabel('Y-axis')
bar_ax.bar(names, values, color='green')
bar_ax.set_title('Bar Plot')
bar_ax.set_xlabel('Fruits')
bar_ax.set_ylabel('Quantity')
plt.tight_layout()
plt.show()

