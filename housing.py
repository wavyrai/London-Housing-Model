## Author: Thijs Verreck
## Importing all the necessary libraries

import numpy as np # For the linear algebra
import pandas as pd # Pandas, what else? 
from pandas.plotting import scatter_matrix # For the scatter matrix 
import matplotlib.pyplot as plt
# Set the fonts for the plots
# If you dont have the fonts installed, you can replace them with any other font you have installed. 
bodyfont = {'fontname':'Helvetica'} # Body font 
titlefont = {'fontname':'Futura'} # Title font (Futura is a premium font, you can comment this line and uncomment the next line if you dont have it installed)
# titlefont = {'fontname':'Helvetica'} # Uncomment this line if you dont have the Futura font installed 

import seaborn as sns # For the heatmap
from IPython.display import HTML
import warnings 
warnings.filterwarnings('ignore') # get rid of annoying warnings
import plotly.express as px # For the interactive map, allows you to zoom in and out (and export to JSON)
import geopandas as gpd # For the map

# Importing the necessary libraries for the machine learning part
# Most come from the scikit-learn library
from sklearn.model_selection import GridSearchCV
from  sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve  


## Importing the data

# Read the data
london_monthly = pd.read_csv('housing_in_london_monthly_variables.csv', parse_dates = ['date']) # The data set is from Justin Cirtautas. Link in the README.md file.

# Print the first five rows of the dataset
print ('This dataset contains {} rows and {} columns.'.format(london_monthly.shape[0], london_monthly.shape[1]))
london_monthly.head(5)

# Print the data types of each column
london_monthly.info()

# Print the number of missing values in each column
london_monthly_zero = london_monthly.isnull().sum().sort_values(ascending = False)
percent = (london_monthly.isnull().sum()/london_monthly.isnull().count()).sort_values(ascending = False)*100
london_monthly_zero = pd.concat([london_monthly_zero, percent], axis = 1, keys = ['Counts', '% Missing'])
print ('These columns have missing data: ')
london_monthly_zero.head(5) #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

# get rid off the 'no_of_crimes' column, because it has too many missing values
london_monthly.drop('no_of_crimes', axis = 1, inplace = True)   

# Fill the missing values in the 'houses_sold' column with the mean of the area
london_monthly['houses_sold'].fillna(london_monthly.groupby('area')['houses_sold'].transform('mean'), inplace = True)

# Extract the year from the date column
london_monthly['year'] = london_monthly['date'].dt.year
london_monthly.iloc[[0, -1]]

# discard the data from 2020, because the year is not complete
london_monthly = london_monthly[london_monthly['year'] < 2020]

# Check the max value of the year column (should be 2019)
london_monthly['year'].max()

# Read the shapefile data into a GeoDataFrame 
london_map = gpd.read_file('London_Wards/Boroughs/London_Borough_Excluding_MHW.shp')
london_boroughs = london_monthly[london_monthly['borough_flag'] == 1]['area'].unique()
len(london_boroughs)

# What are 33 London boroughs? Let's print them with a loop. 
for i, borough in enumerate(london_boroughs):
  print(i+1,':', borough)

# Let's also check the areas that are not London boroughs
print(london_monthly[london_monthly['borough_flag'] == 0]['area'].nunique()) 
print(london_monthly[london_monthly['borough_flag'] == 0]['area'].unique())

# Let's get rid of the non-english regions. 
england_regions = ['east midlands', 'yorks and the humber', 'south west', 'south east', 'east of england', 'west midlands', 'north west', 'north east']
london = london_monthly[london_monthly['area'].isin(london_boroughs)]
england = london_monthly[london_monthly['area'].isin(england_regions)]

# Calculate the average price for each area for each date. 
london_price = london.groupby('date')['average_price'].mean()
england_price = england.groupby('date')['average_price'].mean()

# Plot the time evolution of the average house price in London and England
plt.figure(figsize = (10, 5))
font_size = 14
london_price.plot(y = 'average_price', color = 'navy', lw = 2, label = 'London')
england_price.plot(y = 'average_price', color = 'darkorange', lw = 2, label = 'England')
plt.axvspan('2007-07-01', '2009-06-21', alpha = 0.5, color = '#E57715')
plt.text(x = '2008-05-01', y = 390000, s = 'The GFC', rotation = 90, fontsize = font_size-2, **bodyfont)
plt.axvline(x = '2016-06-23', lw = 2, color = 'darkblue', linestyle = '--')
plt.text(x = '2015-11-01', y = 210000, s = 'Brexit Referendum', rotation = 90, fontsize = font_size-2, **bodyfont)
plt.title('Time evolution of the average house price', size = font_size, **titlefont)
plt.ylabel('Average Price', size = font_size, **bodyfont)
plt.xticks(size = font_size - 3, **bodyfont)
plt.xlabel('Date', size = font_size, **bodyfont)
plt.yticks(size = font_size - 3, **bodyfont)
plt.legend(fontsize = font_size - 3, loc = 'lower right')
plt.savefig('time_evolution_of_average_house_price.jpg', dpi=300) 

# Also wanted to try some Plotly plots, because they have interactive abilities. Feel free to try them out. 
test = px.line(london_monthly, x="date", y="average_price", color='area', template='simple_white', title='Average house price (1995 - 2020)', width=700, height=500)
test.update_layout(title_font_family='Futura', xaxis=dict(title_font_family='Futura'), yaxis=dict(title_font_family='Futura'))
test.show()
test.write_json('fig2.json')
# and a boxplot
fig = px.box(london_monthly, x="average_price", template='simple_white', title='Average house price (1995 - 2020)',
             width=700, height=500, color='borough_flag')
fig.update_layout(title_font_family='Futura', xaxis=dict(title_font_family='Futura'), yaxis=dict(title_font_family='Futura'))
fig.write_json('fig1.json')
fig.show()

# The average price in each borough fluctuates through time. 
# However, we can calculate its mean which can give us a rough indication of how expensive each area is.

london_borough_prices = london.groupby('area')['average_price'].mean()
london_top5_prices = london_borough_prices.sort_values(ascending = False).to_frame()

# Plot the average price in the 5 most expensive London boroughs
london_top5_prices.head(5).sort_values(by = 'average_price', ascending = True).plot(kind = 'barh', figsize = (10, 5), 
                                                                               color = 'bisque', edgecolor = 'sandybrown',
                                                                               legend = False)

plt.title('Average price in the 5 most expensive London boroughs (1995-2019)', size = font_size, y = 1.05, **titlefont)
plt.ylabel('London Borough', size = font_size, **bodyfont)
plt.yticks(size = font_size - 3, **bodyfont)
plt.xlabel('Average Price', size = font_size, **bodyfont)
plt.xticks([0, 200_000, 400_000, 600_000], size = font_size - 3, **bodyfont);
plt.savefig('average_price_in_the_5_most_expensive_london_boroughs.jpg', dpi=300) 
top5 = london_top5_prices.head().index


# Plot the average price in the 5 most expensive London boroughs over time
colors = sns.color_palette("icefire") # I like this color palette :)
plt.figure(figsize = (10, 5))

for index, i in enumerate(top5):
    df = london[london['area'] == i]
    df = df.groupby('date')['average_price'].mean()
    
    df.plot(y = 'average_price', label = i, color = colors[index])
       
plt.title('Average price in the most expensive boroughs', y = 1.04, size = font_size, **titlefont)
plt.xlabel('Date', size = font_size, **bodyfont)
plt.xticks(size = font_size - 3, **bodyfont)
plt.ylabel('Average Price', size = font_size, **bodyfont)
plt.yticks([0.2*1E+6, 0.6*1E+6, 1.0*1E+6, 1.4*1E+6], size = font_size - 3)
plt.legend(fontsize = font_size - 5);
plt.savefig('average_price_in_the_most_expensive_boroughs.jpg', dpi=300) 

# Let's do the same thing for the most expensive English regions
england_prices = england.groupby('area')['average_price'].mean()
england_top3_prices = england_prices.sort_values(ascending = False).to_frame()
top3 = england_top3_prices.head(3).index
colors = sns.color_palette("icefire")

plt.figure(figsize = (10, 5))
for index, i in enumerate(top3):
    df = england[england['area'] == i]
    df = df.groupby('date')['average_price'].mean()
    df.plot(y = 'average_price', label = i, color = colors[index])

plt.title('Average price in the most expensive English regions by date', size = font_size, y = 1.04, **titlefont)
plt.xlabel('Date', size = font_size)
plt.xticks(size = font_size - 3)
plt.ylabel('Average Price', size = font_size)
plt.yticks([100_000, 200_000, 300_000], size = font_size - 3)
plt.legend(fontsize = font_size - 3);
plt.savefig('average_price_in_most_expensive_english_regions_by_date.jpg', dpi=300) 

# And now let's compare the cheapest London borough with the most expensive English regions

plt.figure(figsize = (10, 5))
for index, i in enumerate(top3):
    df_ = england[england['area'] == i]
    df_ = df_.groupby('date')['average_price'].mean()
    df_.plot(y = 'average_price', label = i, color = colors[index], lw = 2, linestyle = '-')

london_barking_price = london[london['area'] == 'barking and dagenham'].groupby('date')['average_price'].mean()
london_barking_price.plot(y = 'average_price', lw = 2, linestyle = '--', color = '#A30015', label = 'Barking and Dagenham')

plt.title('The 3 most expensive English areas + Barking and Dagenham', size = font_size, y = 1.06, **titlefont)
plt.xlabel('Date', size = font_size, **bodyfont)
plt.xticks(size = font_size - 3, **bodyfont)
plt.ylabel('Average Price', size = font_size, **bodyfont)
plt.yticks([0.1*1E+6, 0.2*1E+6, 0.3*1E+6], size = font_size - 3, **bodyfont)
plt.legend(labels = ['South East (Eng)', 'East of England (Eng)', 'South West (Eng)', 'Barking and Dagenham (L)'], 
           fontsize = font_size - 3);
plt.savefig('3_most_expensive_eng+barkinganddagenham.jpg', dpi=300) 
plt.show

# Let's see how the number of houses sold in London has changed over time 
plt.figure(figsize = (10, 5))
london_houses = london.groupby('date')['houses_sold'].sum()
london_houses.plot(figsize = (9, 5), lw = 2, y = 'houses_sold', color = '#00072D')

plt.axvspan('2007-12-21', '2009-06-21', alpha = 0.5, color = '#F08700')
plt.text(x = '2008-04-01', y = 10700, s = 'Recession', rotation = 90, fontsize = font_size-2, **bodyfont)
plt.axvspan('2016-01-1', '2016-05-01', alpha = 0.7, color = '#FFCAAF')
plt.text(x = '2016-06-01', y = 10000, s = 'New Tax Legislation', rotation = 90, fontsize = font_size-2, **bodyfont)

plt.title('Houses sold in London by date', size = font_size, **titlefont)
plt.xlabel('Date', size = font_size, **bodyfont)
plt.xticks(size = font_size - 3, **bodyfont)
plt.ylabel('Houses sold', size = font_size, **bodyfont)
plt.yticks([4000, 8000, 12000, 16000], size = font_size - 3, **bodyfont)
plt.savefig('houses_sold_lnd_date.jpg', dpi=300) 

london_borough_houses = london.groupby('area')['houses_sold'].sum()
london_top5_houses = london_borough_houses.sort_values(ascending = False).to_frame()
london_top5_houses.head(5)

## Time for some maps!

# Read the shapefile data into a GeoDataFrame 
london_map = gpd.read_file('London_Wards/Boroughs/London_Borough_Excluding_MHW.shp')

# Convert the column names to lowercase
london_map.columns = london_map.columns.str.lower()

# Print the first few rows of the GeoDataFrame
print(london_map.head()) 

# Getting ready to merge the dataframes
london_map['name'] = london_map['name'].str.lower()
london_map.rename(columns = {'name': 'area'}, inplace = True)
london_map.rename(columns = {'gss_code': 'code'}, inplace = True)

london_map = london_map[['area', 'code', 'hectares', 'geometry']]
london_map.head()

london_geo_data = london.groupby('area').agg({'average_price': ['mean'], 'houses_sold': 'sum'})

london_geo_data.columns = ['average_price', 'houses_sold']
london_geo_data.reset_index(inplace = True)
london_geo_data.head()

# Check if the areas in the two dataframes are the same
np.intersect1d(london_map['area'], london_geo_data['area']).size

# Merge the two dataframes
london_map = pd.merge(london_map, london_geo_data, how = 'inner', on = ['area'])
london_map.head()

fig, htmap = plt.subplots(1, 2, figsize = (15, 12))

london_map.plot(ax = htmap[0], column = 'average_price', cmap = 'Oranges', edgecolor = 'maroon', legend = True, legend_kwds = {'label': 'Average Price', 'orientation' : 'horizontal'})

london_map.plot(ax = htmap[1], column = 'houses_sold', cmap = 'Greens', edgecolor = 'maroon', legend = True, legend_kwds = {'label': 'Houses Sold', 'orientation' : 'horizontal'})

htmap[0].axis('off')
htmap[0].set_title('Average House Price (All years)', size = font_size, **titlefont)
htmap[1].axis('off')
htmap[1].set_title('Houses Sold (All years)', size = font_size, **titlefont);

plt.savefig('geomap.jpg', dpi=300) 

london_yearly = pd.read_csv('housing_in_london_yearly_variables.csv', parse_dates = ['date'])
london_yearly = london_yearly[london_yearly['area'].isin(london_boroughs)] # select only London boroughs

print ('This dataset contains {} rows and {} columns.'.format(london_yearly.shape[0], london_yearly.shape[1]))
london_yearly.head()

null_london_yearly = london_yearly.isnull().sum().sort_values(ascending = False)
percent = (london_yearly.isnull().sum()/london_yearly.isnull().count()).sort_values(ascending = False)*100

null_df = pd.concat([null_london_yearly, percent], axis = 1, keys = ['Counts', '%'])
null_df.head(10)

london_yearly[~london_yearly['mean_salary'].str.isnumeric()]['mean_salary'].value_counts()

london_yearly['mean_salary'] = london_yearly['mean_salary'].replace(['#'], np.NaN)
london_yearly['mean_salary'] = london_yearly['mean_salary'].astype(float)

london_yearly['year'] = london_yearly['date'].dt.year

print ('yearly_variables dataset')
print ('\tFirst date: ', london_yearly['year'].min())
print ('\tFinal date: ', london_yearly['year'].max())

london_monthly_grouped = london.groupby(['area', 'year']).mean().reset_index()  # group based on area and year (take mean)
london_monthly_grouped = london_monthly_grouped[london_monthly_grouped['year'] >= 1999]            # select all years after 1999 (included)

print ('monthly_variables dataset')
print ('\tFirst date: ', london_monthly_grouped['year'].min())
print ('\tFinal date: ', london_monthly_grouped['year'].max())

london_yearly_group = london_yearly.groupby(['area', 'year']).mean().reset_index() # group it based on area and year
london_yearly_group.head()

london_total = pd.merge(london_yearly_group, london_monthly_grouped, on = ['area', 'year'], how = 'left')
london_total.drop(['borough_flag_x', 'borough_flag_y'], axis = 1, inplace = True)

london_total.head()

corr_table = london_total.corr()
corr_table['average_price'].sort_values(ascending = False)

# Creating a heatmap of the correlations

plt.figure(figsize = (15, 11))

mask = np.triu(np.ones_like(corr_table, dtype = np.bool))

htmap = sns.heatmap(corr_table, mask = mask, annot = True, cmap = 'rocket')
htmap.set_title('Heatmap of pairwise correlations', size = font_size, **titlefont)
htmap.set_xticklabels(htmap.get_xticklabels(), rotation = 45, horizontalalignment = 'right', size = font_size - 3, **bodyfont)
htmap.set_yticklabels(htmap.get_yticklabels(), rotation = 0, horizontalalignment = 'right', size = font_size - 3, **bodyfont);
bottom, top = htmap.get_ylim()
htmap.set_ylim(bottom + 0.5, top - 0.5);

# Saving the heatmap, with 300 dpi. This argument is optional, but it will make the image look better.

plt.savefig('heatmap_of_pairwise_corr.jpg', dpi=300) 

columns = ['average_price', 'median_salary', 'mean_salary', 'number_of_jobs']

scatter_matrix(london_total[columns], figsize = (15, 15), color = '#D52B06', alpha = 0.3, 
               hist_kwds = {'color':['bisque'], 'edgecolor': 'firebrick'}, s = 100, marker = 'o', diagonal = 'hist');

## ML modelling 
# Preparing the data and splitting it into train and test sets

def preparing_data_for_ML(df = london_monthly, training_size = 0.75):
  # Drop unneccessary features, in this case 'borough_flag' and 'code' and 'houses_sold'. 
  # 'borough_flag' is a flag that indicates whether the area is a London borough or not.
  # 'code' is the code of the area.
  # 'houses_sold' is the number of houses sold in the area.
  prepared_df = df.drop(columns =['houses_sold','borough_flag', 'code'])

  # Extract date feature
  prepared_df['year'] = prepared_df['date'].apply(lambda x: x.year)
  prepared_df['month'] = prepared_df['date'].apply(lambda x: x.month)
  prepared_df = prepared_df.drop(columns =['date'])

  # One-hot encoding for area 
  hot_one = pd.get_dummies(prepared_df['area'], drop_first= True)
  prepared_df = pd.concat([prepared_df, hot_one], axis =1)
  prepared_df = prepared_df.drop(columns =['area'], axis =1)
 

  # Given x, y 
  x = prepared_df.drop(columns = ['average_price'])
  y = prepared_df['average_price']


  # Train-test split (train data 75%, test data 25%)
  x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= training_size, shuffle=True, random_state=42)

  # Standard scaling x
  scaler = StandardScaler()
  scaler.fit(x_train)
  x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
  x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

  return x_train, x_test, y_train, y_test

# Use the 'preparing_data_for_ML' function to return x_train, x_test, y_train, y_test 
x_train, x_test, y_train, y_test = preparing_data_for_ML()


## K-Nearest Neighbours
# KNN is a non-parametric method used for classification and regression.
knn = KNeighborsRegressor()
parameters = {'n_neighbors' : [2, 4, 6, 8], 
               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] ,
               }

grid_knn = GridSearchCV(estimator= knn, param_grid = parameters, cv = 3, n_jobs=-1)
grid_knn.fit(x_train, y_train)

# Print the best hyperparameters:
print("\n =========== (1) ==========" )
print("\n Results from Grid Search for KNN " )
print("\n The best estimator across ALL searched params:\n",grid_knn.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_knn.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_knn.best_params_)


# Fitting with parameters
knn_ft = grid_knn.best_estimator_
knn_ft.fit(x_train, y_train)

# Printing the results
predictions = knn_ft.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)


# Plot predicted vs actual
plt.figure(figsize=(7, 4)) 
plt.scatter(y_test, predictions, color='darkorange', alpha=0.5)
plt.xlabel('Actual Labels', **bodyfont)
plt.ylabel('Predicted Labels', **bodyfont)
plt.title('KNN', fontsize= 20, **titlefont)
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='navy', linewidth=2, label='Regression Line')
plt.savefig('knn.jpg', dpi=300) 
plt.show()

# Plot KNN learning curve
# I use the sklearn learning curve function to plot the learning curve: 
train_sizes, train_scores, test_scores = learning_curve(
    estimator=knn_ft, X=x_train, y=y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(7, 4))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="steelblue")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")
plt.plot(train_sizes, train_scores_mean, 'o-', color="steelblue", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Cross-validation score")
plt.xlabel('Training examples', **bodyfont)
plt.ylabel('Score', **bodyfont)
plt.title('KNN Model Learning Curve', fontsize=20, **titlefont)
plt.legend(loc="best")
plt.savefig('knn_learning.jpg', dpi=300) 
plt.show()




## LightGBM
# LightGBM is a gradient boosting framework that uses tree based learning algorithms.
lgbm = LGBMRegressor()
parameters = {'boosting_type' : ['gbdt', "dart", 'goss'],
               'learning_rate': [0.01, 0.03, 0.1] ,
               'n_estimators' : [100, 500, 1000] }

grid_lgbm = GridSearchCV(estimator=lgbm, param_grid = parameters, cv = 3, n_jobs=-1)
grid_lgbm.fit(x_train, y_train)

# Print the best hyperparameters:
print("\n =========== (2) ==========" )
print("\n Results from Grid Search for LGBM " )
print("\n The best estimator across ALL searched params:\n",grid_lgbm.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_lgbm.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_lgbm.best_params_)

lgbm_ft = grid_lgbm.best_estimator_
lgbm_ft.fit(x_train, y_train)

# Printing the results
predictions = lgbm_ft.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.figure(figsize=(7, 4)) 
plt.scatter(y_test, predictions, color = 'darkorange', alpha=0.5)
plt.xlabel('Actual Labels', **bodyfont)
plt.ylabel('Predicted Labels', **bodyfont)
plt.title('LGBM', fontsize= 20, **titlefont)

# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='navy', linewidth=2, label='Regression Line')
plt.savefig('lgbm.jpg', dpi=300) 
plt.show()


## Plot LGBM learning curve
# I use the sklearn learning curve function to plot the learning curve:
train_sizes, train_scores, test_scores = learning_curve(
    estimator=lgbm_ft, X=x_train, y=y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Creating the plot | learning curve
plt.figure(figsize=(7, 4))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="steelblue")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")
plt.plot(train_sizes, train_scores_mean, 'o-', color="steelblue", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Cross-validation score")
plt.xlabel('Training examples', **bodyfont)
plt.ylabel('Score', **bodyfont)
plt.title('LGBM Learning Curve', fontsize=20, **titlefont)
plt.legend(loc="best")
plt.savefig('lgbm_learning.jpg', dpi=300) 
plt.show()




## Random Forest
# Random Forest is an algorithm that is build on multiple decision trees. It can be used both for classification and regression problems.

rfm = RandomForestRegressor()
parameters = {    'n_estimators' : [None, 100 , 300, 1000], 
                  'max_depth'    : [2, 4, 6, None]
                 }

grid_rfm = GridSearchCV(estimator=rfm, param_grid = parameters, cv = 3, n_jobs=-1)
grid_rfm.fit(x_train, y_train)

# Print the best hyperparameters:
print("\n =========== (3) ==========" )
print("\n Results from Grid Search for RandomForest " )
print("\n The best estimator across ALL searched params:\n",grid_rfm.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_rfm.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_rfm.best_params_)

# Fit the model with the best hyperparameters
rfm_ft = grid_rfm.best_estimator_
rfm_ft.fit(x_train, y_train)

# Printing the results
predictions = rfm_ft.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.figure(figsize=(7, 4)) 
plt.scatter(y_test, predictions, color = 'darkorange', alpha=0.5)
plt.xlabel('Actual Labels', **bodyfont)
plt.ylabel('Predicted Labels', **bodyfont)
plt.title('Random Forest', fontsize= 20, **titlefont)
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='navy', linewidth=2, label='Regression Line')
plt.savefig('random_forest.jpg', dpi=300) 
plt.show()

## Plot Random Forest learning curve
# Same here as above, I use the sklearn learning curve function to plot the learning curve: 
train_sizes, train_scores, test_scores = learning_curve(
    estimator=rfm_ft, X=x_train, y=y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


# Creating the plot | learning curve 
plt.figure(figsize=(7, 4))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="steelblue")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")
plt.plot(train_sizes, train_scores_mean, 'o-', color="steelblue", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Cross-validation score")
plt.xlabel('Training examples', **bodyfont)
plt.ylabel('Score', **bodyfont)
plt.title(' Random Forest Learning Curve', fontsize=20, **titlefont)
plt.legend(loc="best")
plt.savefig('random_forest_learning.jpg', dpi=300) 
plt.show()

# That's the end! Thanks for checking out the code all the way to here. 
# I hope you enjoyed it and learned something new. If you have any questions or suggestions, 
# please do reach out on Twitter! I'm @thijsverreck
#
#
#   _____  _      _   _                                           _    
# /__   \| |__  (_) (_) ___    /\   /\ ___  _ __  _ __  ___   ___ | | __
#   / /\/| '_ \ | | | |/ __|   \ \ / // _ \| '__|| '__|/ _ \ / __|| |/ /
#  / /   | | | || | | |\__ \    \ V /|  __/| |   | |  |  __/| (__ |   < 
#  \/    |_| |_||_|_/ ||___/     \_/  \___||_|   |_|   \___| \___||_|\_\
#                 |__/    
# 
# 
# 
#                 
