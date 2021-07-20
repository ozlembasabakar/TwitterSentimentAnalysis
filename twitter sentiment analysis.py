from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords')

missing_value_formats = ["?", "NA", "nan", ""]
dataset = pd.read_csv('gender-classifier.csv',
                      encoding='latin-1', 
                      na_values=missing_value_formats)


genders = dataset[['gender']]
descriptions = dataset[['description']]
names = dataset[['name']]
tweet_counts = dataset[['tweet_count']]
dates = dataset[['created']]

df = pd.concat([descriptions, names, genders, tweet_counts, dates], 
               axis=1)
df['description'] = df['description'].apply(str)
df.replace(missing_value_formats, " ", inplace=True)

# For testing
# df = df.iloc[:1000, :]        

# Separating genders and cleaning the tweets

female = []
male = []
brand = []

for i in range(len(df)):
    if df['gender'][i] == 'female':
        female.append(df['description'][i])
        corpus_female = []
        for i in range(0, len(female)):
            text = re.sub(
                '@[A-Za-z0-9]+|[^0-9A-Za-z \t]|\w+:\/\/\S+', ' ', female[i])
            text = text.lower().split()
            text = ' '.join(text)
            corpus_female.append(text)

    elif df['gender'][i] == 'male':
        male.append(df['description'][i])
        corpus_male = []
        for i in range(0, len(male)):
            text = re.sub(
                '@[A-Za-z0-9]+|[^0-9A-Za-z \t]|\w+:\/\/\S+', ' ', male[i])
            text = text.lower().split()
            text = ' '.join(text)
            corpus_male.append(text)
    elif df['gender'][i] == 'brand':
        brand.append(df['description'][i])
        corpus_brand = []
        for i in range(0, len(brand)):
            text = re.sub(
                '@[A-Za-z0-9]+|[^0-9A-Za-z \t]|\w+:\/\/\S+', ' ', brand[i])
            text = text.lower().split()
            text = ' '.join(text)
            corpus_brand.append(text)


# Mostly used words from all tweets

words = []
words_list = []
for i in range(len(df)):
    text = re.sub('@[A-Za-z0-9]+|[^0-9A-Za-z \t]|\w+:\/\/\S+',
                  ' ', df['description'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    text = ' '.join(text)
    words.append(text)

    for j in range(len(words)):
        x = words[j]
        x = x.split()
        words_list.extend(x)

from collections import Counter
Counter = Counter(words_list)
most_occur = Counter.most_common(10)

lst = pd.DataFrame(most_occur, columns=['Word', 'Count'])
lst.plot.bar(x='Word', y='Count')
plt.title("Mostly used words from all tweets")
plt.show()

# Mostly used words for each gender

word_list_gender = []
param = corpus_male
for i in range(len(param)):
    x = param[i]
    x = x.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    x = [ps.stem(word) for word in x if not word in set(all_stopwords)]
    word_list_gender.extend(x)

from collections import Counter
Counter = Counter(word_list_gender)
most_occur = Counter.most_common(10)

lst = pd.DataFrame(most_occur, columns=['Word', 'Count'])
lst.plot.bar(x='Word', y='Count')
plt.title("Mostly used words from men's tweets")
plt.show()

# Separating tweets by sentiments

positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

gender = male
corpus = corpus_male
for i in range(len(gender)):
    tweet_list.append(gender[i])
    analysis = TextBlob(gender[i])
    score = SentimentIntensityAnalyzer().polarity_scores(gender[i])
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(gender[i])
        negative += 1

    elif pos > neg:
        positive_list.append(gender[i])
        positive += 1

    elif pos == neg:
        neutral_list.append(gender[i])
        neutral += 1


tweet_list = pd.DataFrame(tweet_list)
tweet_list.drop_duplicates(inplace=True)

neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)


def percentage(part, whole):
    return 100 * float(part)/float(whole)


positive = percentage(positive, len(corpus))
negative = percentage(negative, len(corpus))
neutral = percentage(neutral, len(corpus))
polarity = percentage(polarity, len(corpus))
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')


labels = ['Positive ['+str(positive)+'%]', 'Neutral [' +
          str(neutral)+'%]', 'Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue', 'red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.axis('equal')
plt.show()


tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]


tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(
    lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp


tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]

# Prediction of the class

new_review = str(input('New tweet: '))
analysis = TextBlob(new_review)
score = SentimentIntensityAnalyzer().polarity_scores(new_review)
neg = score['neg']
pos = score['pos']

if neg > pos:
    print('negative')

elif pos > neg:
    print('positive')

elif pos == neg:
    print('neutral')

# Prediction of gender

cv = CountVectorizer()
X = cv.fit_transform(words).toarray()
y = df.iloc[:, 2].values
y = y.reshape(1000, -1)

lst_y =[]
lst_x = []
for i in range(len(y)):
    if y[i] == 'male' or y[i] == 'female':
        lst_y.append(y[i])
        lst_x.append(X[i])
lst_x = np.array(lst_x)   

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lst_y = le.fit_transform(lst_y)
                                                
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(lst_x, lst_y, test_size = 0.20, random_state = 0)        

from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(X_train,y_train)
rfc_prediction = rfc_clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, rfc_prediction)


def predictGender(new_review):
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = rfc_clf.predict(new_X_test)
    print(new_y_pred)

predictGender("like the snow, beautiful! but coldãü. I'm a legal drug dealer. Original,Extraordinary, and Psychic! if you want to know me better? talk to me ;)")

# Visualization with WordCLoud

gender = ''.join(map(str, brand))

from PIL import Image
mask = np.array(Image.open('M2jeo.jpg'))

wordcloud = WordCloud(width = 800, height = 800, 
                      random_state=1, background_color='White', 
                      colormap='rainbow', collocations=False, 
                      stopwords = STOPWORDS, 
                      mask=mask).generate(gender)

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=STOPWORDS,
                      min_font_size=10).generate(gender)

def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud)
    plt.axis("off")

plot_cloud(wordcloud)