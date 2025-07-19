# %%
!pip install gutenbergpy
!pip install pyldavis
!pip install nltk -U
!pip install spacy -U
!pip install gensim

# %%
!pip install pandas

# %%
import os
import nltk
import re
import string
import gensim
import numpy as np
import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser
# %%
# === UPDATED START ===

def read_txt_files_to_df(txt_dir):
    data = []
    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(txt_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append({'filename': filename, 'content': content})
    return pd.DataFrame(data)

# Directory containing your txt files
file_dir = r'D:\Data\CapitalIQ_Transcript\Txt_TestRun_v1'

# Read files into DataFrame
df = read_txt_files_to_df(file_dir)
print(df.head())  # preview

# === UPDATED END ===


# %%
# for tokenization
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('wordnet')

# for stopword removal
from nltk.corpus import stopwords
nltk.download('stopwords')

# for lemmatization and POS tagging
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')

# for LDA
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# for LDA evaluation
import pyLDAvis
import pyLDAvis.gensim_models as gensimvisualize



# %%
# load WordNet POS tags for lemmatization
def wordnet_pos_tags(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# preprocessing function
def txt_preprocess_pipeline(text):
    # text is now a string (not a file handle)
    # standardize text to lowercase
    standard_txt = text.lower()
    # remove multiple white spaces and line breaks
    clean_txt = re.sub(r'\n', ' ', standard_txt)
    clean_txt = re.sub(r'\s+', ' ', clean_txt)
    clean_txt = clean_txt.strip()
    # tokenize text
    tokens = word_tokenize(clean_txt)
    # remove non-alphabetic tokens
    filtered_tokens_alpha = [word for word in tokens if word.isalpha() and not re.match(r'^[ivxlcdm]+$', word)]
    # load NLTK stopword list and add original stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['customer', 'business', 'revenue', 'quarter', 'year', 'actually', 'sale', 'market', 'also', 'million', 'unfortunately', 'data', 'advantage', 'anymore'])
    # remove stopwords
    filtered_tokens_final = [w for w in filtered_tokens_alpha if not w in stop_words]
    # define lemmatizer
    lemmatizer = WordNetLemmatizer()
    # conduct POS tagging
    pos_tags = nltk.pos_tag(filtered_tokens_final)
    # lemmatize word-tokens via assigned POS tags
    lemma_tokens = [lemmatizer.lemmatize(token, wordnet_pos_tags(pos_tag)) for token, pos_tag in pos_tags]
    return lemma_tokens

# file iteration function
def iterate_txt_files(txt_dir):
    texts = []
    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(txt_dir, filename), 'r', encoding='utf-8') as file:
                txt_tokens = txt_preprocess_pipeline(file)
                texts.append(txt_tokens)
    return texts





# %%
# === UPDATED START ===

# Apply preprocessing
df['tokens'] = df['content'].apply(txt_preprocess_pipeline)

# Remove empty docs
original_doc_count = len(df)
df = df[df['tokens'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
filtered_doc_count = len(df)
if filtered_doc_count < original_doc_count:
    print(f"Removed {original_doc_count - filtered_doc_count} empty documents after preprocessing.")

texts = df['tokens'].tolist()
print(texts[:1])

# %%
# === BIGRAM/TRIGRAM UPDATE START ===

from gensim.models import Phrases
from gensim.models.phrases import Phraser

# Set this to 'unigram', 'bigram', or 'trigram'
NGRAM_TYPE = 'bigram'   # Change to 'trigram' or 'unigram' as desired

# Build the bigram and trigram models
bigram = Phrases(texts, min_count=5, threshold=100)
trigram = Phrases(bigram[texts], threshold=100)

bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Apply ngram transformation based on choice
if NGRAM_TYPE == 'bigram':
    texts = make_bigrams(texts)
    print("Bigram transformation applied.")
elif NGRAM_TYPE == 'trigram':
    texts = make_trigrams(texts)
    print("Trigram transformation applied.")
else:
    print("Unigram: no n-gram transformation applied.")

print(texts[:1])

# === BIGRAM/TRIGRAM UPDATE END ===



# %%
# Remove any documents that ended up empty after preprocessing
original_doc_count = len(texts)
texts = [doc for doc in texts if len(doc) > 0]
filtered_doc_count = len(texts)
if filtered_doc_count < original_doc_count:
    print(f"Removed {original_doc_count - filtered_doc_count} empty documents after preprocessing.")



# %%
# load dictionary
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_above = .8, no_below = 5)

# generate corpus as BoW
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus[:1])

from gensim.models import LdaMulticore

# train LDA model
lda_model = LdaMulticore(
    corpus=corpus,
    id2word=dictionary,
    num_topics=15,
    passes=10,
    random_state=42
)
   

# print LDA topics
for topic in lda_model.print_topics(num_topics=15, num_words=10):
    print(topic)

# %%
print("Number of documents in corpus:", len(corpus))

# %%
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

# %%
lda_visual = gensimvisualize.prepare(lda_model, corpus, dictionary, mds='mmds')
pyLDAvis.display(lda_visual)


# %%
# generate document-topic distributions
for i, doc in enumerate(corpus):
    doc_topics = lda_model.get_document_topics(doc)
    print(f"Document {i}: {doc_topics}")

# %%
