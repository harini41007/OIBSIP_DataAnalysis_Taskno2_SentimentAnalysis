#import libraries
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load dataset
df = pd.read_csv('Twitter_Data.csv')
df

print(df.info())
print(df['category'].value_counts())

#identify missing value
df.isnull().sum()

#Remove missing rows
df = df.dropna(subset=['category'])
df = df.dropna(subset=['clean_text'])

#checking missing rows after removing
df.isnull().sum()

#Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https\s+','' ,text)
    text = re.sub(r'[^a-zA-Z\s]','' ,text)
    text = re.sub(r'\s+', ' ', text)
    return text
df['clean_text'] = df['clean_text'].apply(clean_text)

#Convert Text to Numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

#Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Logistics Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Teat with Sample Date
sample = ["hope tuthukudi people would prefer honest well behavednationalist courageous likely minister modi cabinet vote benifit thuthukudi"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
label_map ={1:"Positive", 0:"Neutral",-1:"Negative"}
result =int(prediction[0])
print(f"prediction: {result}- {label_map[result]} ")

#Visualize Sentiment
import matplotlib.pyplot as plt
df['category'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")

plt.show()
