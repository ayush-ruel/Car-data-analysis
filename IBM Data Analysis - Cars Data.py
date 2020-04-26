#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# In[95]:


path = "Downloads/imports-85.data"
df = pd.read_csv(path, header=None)


# In[96]:


df.head()


# In[98]:


# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers


# In[99]:


df.head()


# In[100]:


#Saving the data file as CSV
df.to_csv("Downloads/automobile_data.csv", index=False)


# In[101]:


#Cheking the data types of each variable
#As you can see variables like horsepower and price are of the type object which sould be of type int and/or float
#we will fix all this later
df.dtypes


# In[102]:


df.describe(include = "all")


# In[103]:


df[['length', 'compression-ratio']].describe()


# In[104]:


df.info


# In[105]:


df.head(10)


# In[106]:


# As we can that there are missing values that contain "?" 
# So we wil be replacing all those to NaN
df.replace("?", np.nan, inplace = True)


# In[107]:


df.head(10)


# In[108]:


# Checking all the missing values
missing_data = df.isnull()


# In[109]:


# "True" stands for missing value, while "False" stands for not missing value
missing_data.head()


# In[110]:


# Using a for loop to find missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


# Based on the summary above, each column has 205 rows of data, seven columns containing missing data:
# <ol>
#     <li>"normalized-losses": 41 missing data</li>
#     <li>"num-of-doors": 2 missing data</li>
#     <li>"bore": 4 missing data</li>
#     <li>"stroke" : 4 missing data</li>
#     <li>"horsepower": 2 missing data</li>
#     <li>"peak-rpm": 2 missing data</li>
#     <li>"price": 4 missing data</li>
# </ol>
# Now, we will be handling each one of them differently.

# In[111]:


# Using a one liner to change the dtype of "normalized-losses" and replacing all NAN values with the coulmn mean
df["normalized-losses"].replace(np.nan, df["normalized-losses"].astype("float").mean(axis=0), inplace=True)


# In[112]:


# Doing the same with the column "bore"
df["bore"].replace(np.nan, df['bore'].astype('float').mean(axis=0), inplace=True)


# In[113]:


# The same for "stroke", "horsepower" and "peak-rpm"
df["stroke"].replace(np.nan, df['stroke'].astype('float').mean(), inplace=True)


# In[114]:


df["horsepower"].replace(np.nan, df['horsepower'].astype('float').mean(), inplace=True)
df["peak-rpm"].replace(np.nan, df['peak-rpm'].astype('float').mean(), inplace=True)


# In[115]:


df['num-of-doors'].value_counts()


# <ul>
#             <li>We will be replacing missing values with most frequent door type. Because 84% sedans is four doors. Since it is most frequent, hence, most likely to occur.</li>
#         </ul>

# In[116]:


df['num-of-doors'].value_counts().idxmax()


# In[117]:


# Replacing missing data with "four"
df["num-of-doors"].replace(np.nan, "four", inplace=True)


# <ul>
#             <li>In price column there are 4 missing data, simply delete the whole row because price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us</li>
#         </ul>

# In[118]:


# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped four rows
df.reset_index(drop=True, inplace=True)


# In[119]:


df.head(10)


# In[120]:


df.dtypes


# In[121]:


#Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# In[122]:


df.dtypes


# <h3>Data Standardization</h3>
# We will need to apply data transformation to transform mpg into L/100km

# In[123]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]


# In[124]:


# Similary with "highway-mpg" 
df['highway-L/100km'] = 235/df["highway-mpg"]
#df.rename(columns={'highway-mpg': 'highway-L/100km'}, inplace=True)


# In[125]:


df.head()


# <h3>Data Normalization</h3>
# Normalization is the process of transforming values of several variables into a similar range.

# In[126]:


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['length'].max()


# In[127]:


df.head()


# <h3>Data Binnig</h3>
# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.

# In[128]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# In[129]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(df["horsepower"])
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower bins")


# We would like 3 bins of equal size bandwidth

# In[130]:


# We are using numpy's linspace function
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[131]:


# Set group names
group_names = ['Low', 'Medium', 'High']


# In[132]:


#We apply the function "cut" the determine what each value of "df['horsepower']" belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# In[133]:


# The no. of cars in each group
df["horsepower-binned"].value_counts()


# In[134]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(group_names, df["horsepower-binned"].value_counts())

plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower bins")


# In[135]:


# visualizing the bins
get_ipython().run_line_magic('matplotlib', 'inline')

a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower bins")


# we can use categorical variables for regression analysis later. We will use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type.

# In[136]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[137]:


dummy_variable_1.rename(columns={'fuel-type-gas':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()


# In[138]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[139]:


df.head(10)


# In[140]:


# Doing the same for column 'aspiration'
df = pd.concat([df, pd.get_dummies(df["aspiration"])], axis=1)

# drop original column "fuel-type" from "df"
df.drop("aspiration", axis = 1, inplace=True)


# Finally, we have a clean csv file. So, let's export it.

# In[141]:


df.to_csv('Downloads/clean_automobile_df.csv')


# <h3>Now, we will be running EDA on the dataset

# In[142]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[143]:


print(df.dtypes)


# In[144]:


df.corr()


# In[145]:


df[['bore','stroke' ,'compression-ratio','horsepower']].corr()


# In[146]:


# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.title("Relation of Engine size with Price")


# In[147]:


df[["engine-size", "price"]].corr()


# As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.
# 
#  We can examine the correlation between 'engine-size' and 'price' and see it's approximately  0.87
#  
#  Now since Highway mpg is a potential predictor variable of price -

# In[150]:


sns.regplot(x="highway-mpg", y="price", data=df)
plt.xlabel("Highway mpg")
plt.ylabel("Price")
plt.title("Relation of Highway mpg with Price")


# As the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. Highway mpg could potentially be a predictor of price.

# In[151]:


df[['highway-mpg', 'price']].corr()


# In[152]:


sns.regplot(x="peak-rpm", y="price", data=df)


# Since the regression line is almost a horizontal line. Therefore, "Peak-rpm" does not seem like a good predictor of price.

# In[153]:


df[['peak-rpm','price']].corr()


# In[154]:


df[["stroke","price"]].corr()


# In[155]:


#correlation results between "price" and "stroke"
sns.regplot(x="stroke", y="price", data=df)


# In[156]:


sns.boxplot(x="body-style", y="price", data=df)


# We see that the distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price.

# In[157]:


sns.boxplot(x="engine-location", y="price", data=df)


# Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.

# In[158]:


sns.boxplot(x="drive-wheels", y="price", data=df)


# Drive wheels can also be a potential predictor for price.

# <h2>Descriptive Statistical Analysis

# In[159]:


df.describe()


# In[160]:


df.describe(include=['object'])


# In[161]:


df['drive-wheels'].value_counts()


# In[162]:


# We can also convert the above series into dataframe
df['drive-wheels'].value_counts().to_frame()


# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column  'drive-wheels' to 'value_counts'.

# In[163]:


drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


# In[164]:


#rename the index to 'drive-wheels
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# In[165]:


# We can do the for with engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


# Examining the value counts of the engine location would not be a good predictor variable for the price. This is because we only have three cars with a rear engine and 198 with an engine in the front, this result is skewed. Thus, we are not able to draw any conclusions about the engine location.

# <h2>Grouping

# In[166]:


df['drive-wheels'].unique()


# In[170]:


#If we want to know, on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them.
df_group_one = df[['drive-wheels','price']]


# In[171]:


# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one


# In[172]:


# grouping results when grouped by "drive-wheels" and "body-style"
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


# This grouped data is much easier to visualize when it is made into a pivot table.

# In[173]:


grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


# In[174]:


# We can have missing data at some cells of a pivot table
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot


# In[175]:


df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
grouped_test_bodystyle


# Using a heat map to visualize the relationship between Body Style vs Price.

# In[176]:


#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# In[177]:


#Since the default labels does not help much
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# <h2>Correlation

# In[ ]:




