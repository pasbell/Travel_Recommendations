# Travel-Buddy

For this project we created a tool that helps you plan your next trip. Do you have a break
coming up, with no idea where to go? Visit our web app, enter your favorite three city
destinations, your favorite two restaurants in your current city (New York) and our app will
recommend a new city for you to visit, three restaurants in that city, and a list of
activities/attractions we think you would enjoy in that city.


## Datasets
### City Dataset
A dataset of the city features and the counts for each feature - final_table_exp2.csv.
This dataset was compiled by combining:
* A list of 99 cities from by scraping https://www.listchallenges.com/print-list/41692
where we got the names of the cities and their corresponding countries.
* London Cities database which was scraped from
https://data.london.gov.uk/download/global-city-data/ffcefcba-829c-4220-911fd4bf17ef75d6/global-city-indicators.xlsx and returned cities along with their features.
* New York Cities database which was scraped from
http://www.worldcitiescultureforum.com/data and returned cities along with their
features.
* Trip advisor scraping where we first got a list of links corresponding to each city’s page
on Tripadvisor, then iteratively scraped each link for the features of each city returning
each feature and its corresponding numerical value.

We combined all theses datasets by grouping them by city, creating the final dataset
final_table_exp2.csv. 



### Links Dataset
A dataset of the links to each of the tripadvisor features of each city - links_df.csv
Since our cities dataset could only include the features and numerical values in order to run our
machine learning algorithm, a parallel dataset was also generated where the the link from
which the feature was pulled is stored under that feature corresponding to each city. 


### Restaurants Dataset
A dataset that contains data on the first 150 restaurant results for each of
our chosen cities from the Yelp API. - all_restaurants.csv
This dataframe was compiled by using the Yelp API:

* The rating and categories of each restaurant were retrieved from the yelp API and
binary variables were used to determine whether a restaurant belonged to a category
* The top 3 reviews of each restaurant (since this is all the yelp API allows) were compiled
into one text document and a sentiment analysis was run. Each emotion was used as an
attribute in the table. 

## Methodology
The user inputs 3 cities and 2 restaurants.

An unsupervised machine learning algorithm is used on the Cities Dataset to find a
city recommendation, a city most similar to the three cities entered by the user, in terms of
their features. This is done using the unsupervised nearest neighbors machine learning
algorithm. To do this, we first scaled the data using StandardScaler. Then the data was fit to a
nearest neighbor model using the 3 input cities to find the distances between each of these
cities and the rest. These distances were summed over the three input cities and then the
dataset was reordered according the the closest distance. The top result, not including the
input cities, was chosen as the recommended city.

A similar unsupervised machine learning algorithm is used on the restaurants
dataset to find a three restaurants in the recommended city, those that are most similar to the
two restaurants entered by the user, in terms of their categories and reviews sentiments. This is
done by first using the Yelp API to search for the attributes of the entered cities in New York and conducting a sentiment analysis on the reveiws. Then, an unsupervised nearest neighbors ML alrgorithm was used and, filtering on the restaurants in the city that had been recommended, the top 3 results were selected.

Finally, another function uses the three city entries and the Cities dataset to
find the three most common features among the entered cities. This is done by summing the distance 
values of the features of these three cities and selecting the top three features. If these
features apply to the recommended city, the function then finds the links to the pages where
each of the features was found and grabs the top three headlines from each page and returns
them as the recommended activities with links to them on tripadvisor.
