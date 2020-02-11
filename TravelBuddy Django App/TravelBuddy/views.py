from django.shortcuts import render
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from TravelBuddy.models import City, Attraction, Restaurant


# your views here.
def home(request):
    return render(request, "home.html")


def results(request):

    city1 = request.POST.get('city1', None)
    city2 = request.POST.get('city2', None)
    city3 = request.POST.get('city3', None)
    rest1 = request.POST.get('restaurant1', None)
    rest2 = request.POST.get('restaurant2', None)


    final_table_exp2 = pd.DataFrame(list(City.objects.all().values()))
    links_df = pd.DataFrame(list(Attraction.objects.all().values()))
    all_restaurants = pd.DataFrame(list(Restaurant.objects.all().values()))

    links_df.drop(columns='id', inplace=True)
    all_restaurants.drop(columns='id', inplace=True)
    final_table_exp2.drop(columns='id', inplace=True)
    links_df = links_df.set_index('city')


    recommended_city = city_ml(city1, city2, city3, final_table_exp2)
    #attractions_df = get_recs(recommended_city,city1,city2,city3,links_df,final_table_exp2)
    print(recommended_city)
    print(links_df.head())

    recommended_rests=recommend_restaurants(recommended_city, 3, [rest1, rest2], all_restaurants)
    print(recommended_rests)


    context = dict()
    context['city_name'] = recommended_city
    context['restaurant_names'] = recommended_rests #["rest1", "rest2", "rest3"]
    context['attraction_names'] = ["Museum", "Park"] #recommended_attracts
    context['links'] = ["link1", "link2"]  #recommended_attracts

    return render(request, "results.html", context)


#Code
###ML
def city_ml(city1, city2, city3, final_table_exp2):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    scaler = StandardScaler()
    scaler.fit(final_table_exp2.iloc[:, 1:])
    final_table_exp21 = scaler.transform(final_table_exp2.iloc[:, 1:])

    index1 = final_table_exp2[final_table_exp2["city"] == city1].index
    index2 = final_table_exp2[final_table_exp2["city"] == city2].index
    index3 = final_table_exp2[final_table_exp2["city"] == city3].index

    city_indices = [index1[0], index2[0], index3[0]]

    N = final_table_exp2.shape[0]
    model = NearestNeighbors(n_neighbors=N)
    model.fit(final_table_exp21)
    distances, indices = model.radius_neighbors(final_table_exp21[city_indices, :], radius=np.inf)

    distance = np.sum(distances)
    indices_reordered = np.argsort(distance)

    recommendations = final_table_exp2.iloc[indices_reordered, :1].drop(city_indices)
    return recommendations.iloc[0].values[0]


def recommend_restaurants(city, number_rests, rest_names, all_restaurants):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    for rest_name in rest_names:
        new_rest = get_restaurant_attributes('New York', rest_name)
        new_rest.rename(index={0: all_restaurants.shape[0]}, inplace=True)
        all_restaurants = all_restaurants.append(new_rest)

    all_restaurants = all_restaurants.fillna(0)

    restaurant_indices = np.arange(all_restaurants.shape[0] - (len(rest_names) - 1) - 1, all_restaurants.shape[0])

    not_features = ['city', 'name', 'latitude', 'longitude']
    rest_cols = all_restaurants.columns.values
    city_index = np.where(rest_cols == 'city')[0][0]
    name_index = np.where(rest_cols == 'name')[0][0]
    latitude_index = np.where(rest_cols == 'latitude')[0][0]
    longitude_index = np.where(rest_cols == 'longitude')[0][0]

    not_feature_indices = [city_index, name_index, latitude_index, longitude_index]
    features = np.delete(all_restaurants.columns.values, not_feature_indices)

    scaler = StandardScaler()
    scaler.fit(all_restaurants.loc[:, features])
    restaurants = scaler.transform(all_restaurants.loc[:, features])

    N = all_restaurants.shape[0]
    model = NearestNeighbors(n_neighbors=N)
    model.fit(restaurants)
    distances, indices = model.radius_neighbors(restaurants[restaurant_indices, :], radius=np.inf)
    distance = np.sum(distances)
    indices_reordered = np.argsort(distance)

    rest_recs = all_restaurants.loc[indices_reordered, not_features].drop(restaurant_indices)

    return rest_recs[rest_recs["city"] == city]['name'].values[0:number_rests]


# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'  # The API url header
SEARCH_PATH = '/v3/businesses/search'  # The path for an API request to find businesses
BUSINESS_PATH = '/v3/businesses/'  # The path to get data for a single business
API_KEY = '7sKhsf-uv7HP5UhdD-DRpE6CE0SZeFGk3a7kyZ45-nezAU8cXeSPV1OaF5IMg9e7C5UuuJMJmMeNXBb5ekLpVLZ5061HER6ID0D0poVbccZHSGl7XtOuAwBvLNnnXXYx'


# This function gets the list of businesses near the location
def get_restaurants(api_key, location, number, restaurant):
    try:
        import requests

        offset = 0
        businesses = []
        # First we get the access token
        # Set up the search data dictionary
        search_data = {
            'term': restaurant,
            'location': location.replace(' ', '+'),
            'limit': 50,
            'offset': offset
        }
        url = API_HOST + SEARCH_PATH
        headers = {
            'Authorization': 'Bearer %s' % api_key,
        }

        if number < 50:
            offset_limit = 0
        else:
            offset_limit = number - 50

        while offset <= offset_limit:
            response = requests.request('GET', url, headers=headers, params=search_data).json()

            results = response.get('businesses')
            businesses.extend(results)

            offset += 50
            search_data['offset'] = offset

        return businesses[:number]
    except:
        print("city skipped")
        return list()


# This function gets reviews for each business from yelp
def get_business_review(api_key, business_id):
    import json
    import requests

    try:
        business_path = BUSINESS_PATH + business_id + "/reviews"
        url = API_HOST + business_path

        headers = {
            'Authorization': 'Bearer %s' % api_key,
        }

        response = requests.request('GET', url, headers=headers).json()

        review_text = ''
        for review in response['reviews']:
            review_text += review['text']
        return review_text
    except:
        return ''


# This function gets restaurants (businesses) and then reviews for each restaurant
def get_reviews(location, number, restaurant):
    restaurants = get_restaurants(API_KEY, location, number, restaurant)

    if not restaurants:
        return None
    restaurant_details = list()
    review_list = list()
    for restaurant in restaurants:
        restaurant_name = restaurant['name']
        restaurant_rating = restaurant['rating']
        restaurant_categories = restaurant['categories']
        restaurant_city = restaurant['location']['city']
        restaurant_location = restaurant['coordinates']
        restaurant_id = restaurant['id']

        review_reviews = get_business_review(API_KEY, restaurant_id)
        restaurant_details.append(
            (restaurant_name, restaurant_rating, restaurant_categories, restaurant_city, restaurant_location))
        review_list.append((restaurant_name, review_reviews))
    return restaurant_details, review_list


def get_nrc_data():
    nrc = "/Users/pasbell/Google Drive/College/4th Year/Fall/Data Analytics/Class Notebooks/local_nltk_data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    count = 0
    emotion_dict = dict()
    with open(nrc, 'r') as f:
        all_lines = list()
        for line in f:
            if count < 46:
                count += 1
                continue
            line = line.strip().split('\t')
            if int(line[2]) == 1:
                if emotion_dict.get(line[0]):
                    emotion_dict[line[0]].append(line[1])
                else:
                    emotion_dict[line[0]] = [line[1]]
    return emotion_dict


emotion_dict = get_nrc_data()


def emotion_analyzer(text, emotion_dict=emotion_dict):
    # Set up the result dictionary
    emotions = {x for y in emotion_dict.values() for x in y}
    emotion_count = dict()
    for emotion in emotions:
        emotion_count[emotion] = 0

    # Analyze the text and normalize by total number of words
    total_words = len(text.split())
    for word in text.split():
        if emotion_dict.get(word):
            for emotion in emotion_dict.get(word):
                emotion_count[emotion] += 1 / len(text.split())
    return emotion_count


def comparative_emotion_analyzer(doc_tuples):
    import pandas as pd
    df = pd.DataFrame(columns=['name', 'Anger', 'Fear', 'Trust', 'Negative',
                               'Positive', 'Joy', 'Disgust', 'Anticipation',
                               'Sadness', 'Surprise'], )
    df.set_index("name", inplace=True)

    output = df
    for text_tuple in doc_tuples:
        text = text_tuple[1]
        result = emotion_analyzer(text)
        df.loc[text_tuple[0]] = [result['anger'], result['fear'], result['trust'],
                                 result['negative'], result['positive'], result['joy'], result['disgust'],
                                 result['anticipation'], result['sadness'], result['surprise']]
    return output


def get_restaurant_attributes(city, restaurant):

    # API from Yelp
    restaurant_details, review_list = get_reviews(city, 1, restaurant)

    ##Clean Data

    restaurants_df = pd.DataFrame(restaurant_details, columns=['name', 'rating', 'categories', 'city', 'location'])
    restaurants_df.city = city
    restaurants_df['latitude'] = restaurants_df['location'].apply(lambda x: x['latitude'])
    restaurants_df['longitude'] = restaurants_df['location'].apply(lambda x: x['longitude'])
    restaurants_df = restaurants_df.drop(['location'], axis=1)

    restaurants_df['categories'] = restaurants_df['categories'].apply(lambda x: [y['alias'] for y in x])
    restaurants_df = restaurants_df.set_index('name')

    ##Convert Categories to columns
    restaurants_df['categories'] = restaurants_df['categories'].apply(lambda x: ','.join(x))
    restaurants_df = restaurants_df.merge(restaurants_df['categories'].str.get_dummies(sep=','), left_on="name",
                                          right_on="name")
    restaurants_df = restaurants_df.drop(['categories'], axis=1)
    cols = list(restaurants_df.columns)
    cols = cols[1:] + [cols[0]]
    restaurants_df = restaurants_df[cols]
    restaurants_df = restaurants_df.reset_index()

    # Add Review Emotions
    restaurants_df = restaurants_df.merge(comparative_emotion_analyzer(review_list), left_on="name", right_on="name")

    ##Add to dataframe
    return restaurants_df


def get_recs(city_rec, city1, city2, city3, links_df, final_table_exp2):
    links_dict = links_df.to_dict('index').get(city_rec)
    links = {k: v for k, v in links_dict.items() if v != 0}

    mask = [x in links.keys() for x in final_table_exp2.columns]
    att_df = final_table_exp2[final_table_exp2.columns[mask]]
    att_df = att_df.set_index(final_table_exp2['city'])
    sums = pd.DataFrame(att_df.T[city1] + att_df.T[city2] + att_df.T[city3])

    sums_sorted = sums.sort_values(by=[0], ascending=False)
    feats = sums_sorted.index.values

    urls = []
    titles = []
    i = 0
    while len(urls) <= 3:
        try:
            response = requests.get(links[feats[i]])
            print(response)
            results = BeautifulSoup(response.content, 'lxml')
            print(results)
            rec = results.find_all('a', {'class': "attractions-ap-product-card-ListingTitle__listingTitle--1v6bA"})
            for i in range(min(len(rec), 3)):
                url = 'https://www.tripadvisor.com' + rec[i].get('href')
                title = rec[i].get_text().strip('\n')
                urls = np.append(urls, url)
                titles = np.append(titles, title)
            response = requests.get(links[feats[i]])
            results = BeautifulSoup(response.content, 'lxml')
            rec = results.find_all('div', {'class': "listing_title title_with_snippets "})
            for i in range(min(len(rec), 3)):
                url = 'https://www.tripadvisor.com' + rec[i].find('a').get('href')
                title = rec[i].find('a').get_text().strip('\n')
                urls = np.append(urls, url)
                titles = np.append(titles, title)
            i = i + 1
        except:
            i = i + 1

    titles = pd.DataFrame(titles).drop_duplicates()
    urls = pd.DataFrame(urls).drop_duplicates()
    rec_df = pd.DataFrame()
    rec_df['title'] = titles[0]
    rec_df['url'] = urls[0]
    return rec_df