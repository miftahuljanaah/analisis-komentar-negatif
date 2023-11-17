
import re
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')

data = pd.read_csv('dataset/dataset.csv', delimiter=';')
df_stopwords = pd.read_csv('stopwords-id.csv', header=None, names=['stopword'])
df_slang = pd.read_csv('kamus-singkatan.csv', delimiter=';', names=['singkatan', 'kata'])
df_lexicon = pd.read_csv('lexicon.csv')
df_corpus = pd.read_csv('corpus.csv')

stemmer = StemmerFactory().create_stemmer()

def preprocessing(text):
    global df_slang, df_stopwords, df_lexicon, df_corpus
    def cleaning(text):
        text = text.replace('-ness', '').replace('-jualness', '')
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'/n', ' ', text)
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'(?<!\bunnes)(\w)(\1+)(?=\s|[\.,!])', r'\1', text)
        text = text.strip(' ')
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = text.lower()  
        return text
    
    def replace_word_elongation(text):
        elongated_words = re.findall(r'\b\w*(?:(\w)\1{2,})\w*\b', text)
        for word in elongated_words:
            replacement = word[0]
            text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text)
        return text
    
    def tokenize(text):
        text = word_tokenize(text)
        return text
    
    def translate_slang_list(text_list):
        global df_slang
        translated_list = []
        for text in text_list:
            words = text.split()
            translated_words = []
            for word in words:
                if word in df_slang['singkatan'].tolist():
                    translated_words.append(df_slang[df_slang['singkatan'] == word]['kata'].values[0])
                else:
                    translated_words.append(word)
            translated_list.append(' '.join(translated_words))
        return text_list
    
    def remove_stopwords(text):
        global df_stopwords
        if isinstance(text, list):
            filtered_words = [word for word in text if word.lower() not in df_stopwords['stopword'].str.lower().values]
            return filtered_words
        else:
            return text
    
    def lemmatization(tokens):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        lemmatized_tokens = [stemmer.stem(token) for token in tokens]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return text
    
    def labelling(text):
        global df_lexicon, df_corpus
        words = text.lower().split()
        score = 0
        for word in words:
            if word in df_lexicon['word'].values:
                weight = df_lexicon.loc[df_lexicon['word'] == word, 'weight'].values[0]
                score += weight
        if score > 0:
            return 'positif'
        elif score < 0:
            service_words = df_corpus['kata'].values
            for service_word in service_words:
                if service_word in text:
                    return 'negatif'
            return 'netral'
        else:
            return 'netral'

    
    text = cleaning(text)
    text = replace_word_elongation(text)
    tokens = tokenize(text)
    tokens = translate_slang_list(tokens)
    tokens = remove_stopwords(tokens)
    lemmatized_text = lemmatization(tokens)
    label = labelling(lemmatized_text)
    return label

data['label'] = data['full_text'].apply(preprocessing)

X = data['full_text']  
y = data['label']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm_model = SVC(kernel='rbf', C=100, gamma=0.01)
svm_model.fit(X_train_tfidf, y_train)

predictions = svm_model.predict(X_test_tfidf)






