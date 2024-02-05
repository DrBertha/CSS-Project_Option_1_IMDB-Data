import pandas as pd
import numpy as np

# Load dataset 
file_path = "/home/manager/CHPC_summer_school/movie_dataset.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())

# Replace spaces in column names with underscores
df.columns = df.columns.str.replace(' ', '_')

# Display the count of missing values in each column
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Basic statistics of numeric columns
print(df.describe())

# Distribution of categorical columns
for column in df.select_dtypes(include='object').columns:
    print(f"\n{column} distribution:\n{df[column].value_counts()}")

# Exclude non-numeric columns when calculating correlation matrix
numeric_columns = df.select_dtypes(include=np.number).columns
correlation_matrix = df[numeric_columns].corr()

# Display the correlation matrix
print("\nCorrelation Matrix:\n", correlation_matrix)

# Assuming the column containing movie ratings is named 'Rating'
highest_rated_movie = df[df['Rating'] == df['Rating'].max()]

# Display the details of the highest-rated movie
print("Highest Rated Movie:\n", highest_rated_movie)

# Filter data for movies released from 2015 to 2017
filtered_data = df[(df['Year'] >= 2015) & (df['Year'] <= 2017)]

# Calculate the average revenue for the filtered data
average_revenue_2015_to_2017 = filtered_data['Revenue_(Millions)'].mean()

# Display the result
print("Average Revenue (2015 to 2017):", average_revenue_2015_to_2017)

# Count the number of movies released in 2016
movies_2016_count = df[df['Year'] == 2016].shape[0]

# Display the result
print("Number of Movies Released in 2016:", movies_2016_count)

# Count the number of movies directed by Christopher Nolan
nolan_movies_count = df[df['Director'] == 'Christopher Nolan'].shape[0]

# Display the result
print("Number of Movies Directed by Christopher Nolan:", nolan_movies_count)

# Count the number of movies with a rating of at least 8.0
high_rated_movies_count = df[df['Rating'] >= 8.0].shape[0]

# Display the result
print("Number of Movies with a Rating of at least 8.0:", high_rated_movies_count)

# Filter the dataset for movies directed by Christopher Nolan
nolan_movies = df[df['Director'] == 'Christopher Nolan']

# Calculate the median rating of movies directed by Christopher Nolan
median_nolan_rating = nolan_movies['Rating'].median()

# Display the result
print("Median Rating of Movies Directed by Christopher Nolan:", median_nolan_rating)

# Group the dataset by 'Year' and calculate the average rating for each year
average_rating_by_year = df.groupby('Year')['Rating'].mean()

# Find the year with the highest average rating
year_highest_avg_rating = average_rating_by_year.idxmax()

# Display the result
print("Year with the Highest Average Rating:", year_highest_avg_rating)

# Filter the dataset for movies released between 2006 and 2016 (inclusive)
movies_2006_to_2016 = df[(df['Year'] == 2006) & (df['Year'] <= 2016)]

# Calculate the number of movies in each year
num_movies_2006 = len(df[df['Year'] == 2006])
num_movies_2016 = len(df[df['Year'] == 2016])
num_movies_2006_to_2016 = len(movies_2006_to_2016)

# Calculate the percentage increase
percentage_increase = ((num_movies_2016 - num_movies_2006) / num_movies_2006) * 100

# Display the result
print("Percentage Increase in Number of Movies (2006 to 2016):", percentage_increase)

# Split the 'Actors' column into individual actors
all_actors = df['Actors'].str.split(', ', expand=True)

# Reshape the dataframe to have a single column with all actors
all_actors_stacked = all_actors.stack()

# Find the most common actor
most_common_actor = all_actors_stacked.value_counts().idxmax()

# Display the result
print("Most Common Actor in All Movies:", most_common_actor)

# Split the 'Genre' column into individual genres
all_genres = df['Genre'].str.split(', ', expand=True)

# Reshape the dataframe to have a single column with all genres
all_genres_stacked = all_genres.stack()

# Count the number of unique genres
unique_genres_count = all_genres_stacked.nunique()

# Display the result
print("Number of Unique Genres in the Dataset:", unique_genres_count)
