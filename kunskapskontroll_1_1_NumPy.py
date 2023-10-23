#!/usr/bin/env python
# coding: utf-8

# # NumPy

# Read the links: https://numpy.org/doc/stable/user/quickstart.html  and https://numpy.org/doc/stable/user/basics.broadcasting.html  before solving the exercises. 

# In[ ]:


import numpy as np


# ### Print out the dimension (number of axes), shape, size and the datatype of the matrix A.

# In[1]:


import numpy as np

A = np.arange(1, 16).reshape(3, 5)
num_axes = A.ndim
shape = A.shape
size = A.size
dtype = A.dtype
print("Number of Axes (Dimension):", num_axes)
print("Shape (Rows, Columns):", shape)
print("Size (Total Number of Elements):", size)
print("Data Type:", dtype)


# ### Do the following computations on the matrices B and C: 
# * Elementwise subtraction. 
# * Elementwise multiplication. 
# * Matrix multiplication (by default you should use the @ operator).

# In[ ]:


B = np.arange(1, 10).reshape(3, 3)
C = np.ones((3, 3))*2

print(B)
print()
print(C)


# In[4]:


elementwise_subtraction = B - C
elementwise_multiplication = B * C
Martix_multiplication = B @ C
print("Martix",B)
print("Martix",C)
print("Elementwise subtraction :",elementwise_subtraction)
print("Elementwise multiplication :",elementwise_multiplication)
print("Martix multiplication :",Martix_multiplication)






# ### Do the following calculations on the matrix:
# * Exponentiate each number elementwise (use the np.exp function).
# 
# * Calculate the minimum value in the whole matrix. 
# * Calculcate the minimum value in each row. 
# * Calculcate the minimum value in each column. 
# 
# 
# * Find the index value for the minimum value in the whole matrix (hint: use np.argmin).
# * Find the index value for the minimum value in each row (hint: use np.argmin).
# 
# 
# * Calculate the sum for all elements.
# * Calculate the mean for each column. 
# * Calculate the median for each column. 

# In[10]:


B = np.arange(1, 10).reshape(3, 3)
print(B)
Exponentiate_matrix = np.exp(B)
MinimumValue_matrix = np.min(B)
MinValueInRow_matrix = np.min(B, axis=1)
MinValueInCol-matrix = np.min(B, axis=0)
IndexMinValue_matrix = np.unravel_index(np.argmin(B), B.shape)
IndexMinValuesPer_row = np.argmin(B, axis=1)
SumOfElements = np.sum(B)
MeanPerColumn = np.mean(B, axis=0)
MedianPerColumn = np.median(B, axis=0)
print("\nMinimum Value in the Whole Matrix (np.min):", MinimumValue_matrix)
print("\nMinimum Value in Each Row (np.min, axis=1):", MinValueInRow_matrix)
print("\nMinimum Value in Each Column (np.min, axis=0):", MinValueInCol_matrix)
print("\nIndex of Minimum Value in the Whole Matrix (np.argmin):", IndexMinValue_matrix)
print("\nIndex of Minimum Value in Each Row (np.argmin, axis=1):", IndexMinValuesPer_row)
print("\nSum of All Elements (np.sum):", SumOfElements)
print("\nMean for Each Column (np.mean, axis=0):", MeanPerColumn)
print("\nMedian for Each Column (np.median, axis=0):", MedianPerColumn)






# ### What does it mean when you provide fewer indices than axes when slicing? See example below.

# In[ ]:


print(A)


# In[ ]:


A[1]


# **Answer:**

# In[ ]:





# ### Iterating over multidimensional arrays is done with respect to the first axis, so in the example below we iterate trough the rows. If you would like to iterate through the array *elementwise*, how would you do that?

# In[ ]:


A


# In[ ]:


for i in A:
    print(i)


# In[ ]:


for row in A:
    for element in row:
        print(element)


# ### Explain what the code below does. More specifically, b has three axes - what does this mean? 

# In[ ]:


a = np.arange(30)
b = a.reshape((2, 3, -1))
print(a)
print()

print(b)


# In[ ]:


a is 1d array[0,29] 
b is 3 d array [2,3,5]


# ### Broadcasting
# **Read the following link about broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html#basics-broadcasting**

# # Remark on Broadcasting when doing Linear Algebra calculations in Python. 

# ### From the mathematical rules of matrix addition, the operation below (m1 + m2) does not make sense. The reason is that matrix addition requires two matrices of the same size. In Python however, it works due to broadcasting rules in NumPy. So you must be careful when doing Linear Algebra calculations in Python since they do not follow the "mathematical rules". This can however easily be handled by doing some simple programming, for example validating that two matrices have the same shape is easy if you for instance want to add two matrices. 

# In[ ]:


m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([1, 1])
print(m1 + m2)


# ### The example below would also not be allowed if following the "mathematical rules" in Linear Algebra. But it works due to broadcasting in NumPy. 

# In[ ]:


v1 = np.array([1, 2, 3])
print(v1 + 1)


# In[ ]:


A = np.arange(1, 5).reshape(2,2)
print(A)

b = np.array([2, 2])
print(b)


# # Linear Algebra Exercises

# The exercies are taken from the "Matrix Algebra for Engineers" by Chasnov: https://www.math.hkust.edu.hk/~machas/matrix-algebra-for-engineers.pdf .
# 
# Do the following exercises: 
# * Chapter 2, exercise 1-3.
# * Quiz on p.8, exercise 2. 
# * Chapter 6, exercise 1. 
# * Quiz on p.15, exercise 3. 
# 
# 
# * Chapter 10, exercise 1. 
# * Chapter 12 exercise 1. 
# 

# In[ ]:


A = np.array([[2, 1, -1], [1, -1, 1]])
B = np.array([[4, -2, 1], [2, -4, -2]])

C = np.array([[1, 2], [2, 1]])
D = np.array([[3, 4], [4, 3]])

E = np.array([[1], [2]])

print(A)
print(B)
print(C)
print(D)
print(E)


# **Chap2. Question 1.**
# 
# **Write a function "add_mult_matrices" that takes two matrices as input arguments (validate that the input are of the type numpy.ndarray by using the isinstance function), a third argument that is either 'add' or 'multiply' that specifies if you want to add or multiply the matrices (validate that the third argument is either 'add' or 'multiply'). When doing matrix addition, validate that the matrices have the same size. When doing matrix multiplication, validate that the sizes conform (i.e. number of columns in the first matrix is equal to the number of rows in the second matrix).**
# 
# In this exercise, create a function that takes two matrices as input and either adds or multiplies them by specifying a argument as either 'add' or 'multiply'. Validate that both matrices taken as input are of the type ndarray (use the isinstance function).

# In[ ]:


import numpy as np

def add_mult_matrices(matrix1, matrix2, operation):
    if not isinstance(matrix1, np.ndarray) or not isinstance(matrix2, np.ndarray):
        raise ValueError("both impute must be numpy.")
    
    if operation == 'add':
        if matrix1.shape != matrix2.shape:
            raise ValueError("Matrice addition.")
        result = matrix1 + matrix2
    elif operation == 'multiply':
        if matrix1.shape[1] != matrix2.shape[0]:
            raise ValueError("Matrices multiplication.")
        result = np.dot(matrix1, matrix2)
    else:
        raise ValueError("Operation must be 'add' or 'multiply'.")
    
    return result

matrix_A = np.array([[1, 2], [3, 4]])
matrix_B = np.array([[5, 6], [7, 8]])

result_add = add_mult_matrices(matrix_A, matrix_B, 'add')
print("Addition Result:\n", result_add)

result_multiply = add_mult_matrices(matrix_A, matrix_B, 'multiply')
print("Multiplication Result:\n", result_multiply)


# **Chap2. Question 2**

# In[ ]:





# **Chap2. Question 3**

# In[ ]:





# **Quiz p.11, Question 2**

# In[ ]:





# **Chap 6. Question 1**

# In[ ]:





# **Quiz p.19, Question 3**

# In[ ]:





# **Chap10. Question 1 a)**

# In[ ]:





# **Chap10. Question 1 b)**

# In[ ]:





# **Chap 12. Question 1**

# In[ ]:





# ### Copies and Views
# Read the following link: https://numpy.org/doc/stable/user/basics.copies.html

# **Basic indexing creates a view, How can you check if v1 and v2 is a view or copy? If you change the last element in v2 to 123, will the last element in v1 be changed? Why?**

# In[ ]:


v1 = np.arange(4)
v2 = v1[-2:]
print(v1)
print(v2)


# In[ ]:


# The base attribute of a view returns the original array while it returns None for a copy.
print(v1.base)
print(v2.base)


# In[ ]:


# The last element in v1 will be changed aswell since v2 is a view, meaning they share the same data buffer.
v2[-1] = 123
print(v1)
print(v2)


# In[ ]:




