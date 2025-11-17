```Python
# Expt. 1: To study and implement word frequency count
#Word Frequency
import re
import nltk
from collections import Counter

# Download POS tagger (only first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def analyze(text):
    # Lowercase
    text_lower = text.lower()

    # Tokenize sentences, lines, word
    sentences = nltk.sent_tokenize(text)
    lines = text.split("\n")
    words = nltk.word_tokenize(text_lower)

     # Extract punctuation tokens
    punctuation = re.findall(r'[^\w\s]', text)

    # Count spaces and tabs
    spaces = re.findall(r' ', text)
    tabs = re.findall(r'\t', text)
    
    # --- UNIQUE SETS ---
    unique_sentences = list(set(sentences))
    unique_lines = list(set([l for l in lines if l.strip() != ""]))
    unique_words = list(set([w for w in words if w.isalnum()]))
    unique_punctuation = list(set(punctuation))

    # --- POS TAGGING ---
    words_with_pos = nltk.pos_tag(words)

    # --- FREQUENCY ---
    word_freq = Counter([w for w in words if w.isalnum()])

    # Bundle everything
    tokenized_data = {
        "sentences": sentences,
        "lines": lines,
        "words": words,
        "spaces": spaces,
        "tabs": tabs,
        "punctuation": punctuation,
    }
    return unique_sentences, unique_lines, unique_words, unique_punctuation, tokenized_data, words_with_pos, word_freq


# Example
sample_text = """
Natural Language Processing is the study of making machines understand human language.
Language models learn patterns in language through data.
This is an extra line.
"""

# Run analysis
(unique_sentences, unique_lines, unique_words,
 unique_punctuation, tokenized_data,
 words_with_pos, word_freq) = analyze(sample_text)

# Results
print("\n=== Unique Tokenization Results ===")
print("Unique Sentences:", unique_sentences)
print("Count of Unique Sentences:", len(unique_sentences))

print("\nUnique Lines:", unique_lines)
print("Count of Unique Lines:", len(unique_lines))

print("\nUnique Words:", unique_words)
print("Count of Unique Words:", len(unique_words))

print("\nUnique Punctuation:", unique_punctuation)
print("Count of Unique Punctuation:", len(unique_punctuation))

print("\n=== Total Token Counts (including duplicates) ===")
print("Total sentences:", len(tokenized_data['sentences']))
print("Total lines:", len(tokenized_data['lines']))
print("Total words:", len(tokenized_data['words']))
print("Total spaces:", len(tokenized_data['spaces']))
print("Total tabs:", len(tokenized_data['tabs']))
print("Total punctuation marks:", len(tokenized_data['punctuation']))

print("\n=== Words with POS Tags (first 10) ===")
print(words_with_pos[:10])

print("\n=== Word Frequency (Top 10) ===")
print(word_freq.most_common(10))
```

```Python
Expt. 2: To implement a python program for case folding, stop word removal and time token ratio
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_and_ttr_nltk(text):
    # --- 1. Case folding ---
    text = text.lower()

    # --- 2. Tokenization ---
    tokens = word_tokenize(text)

    # --- 3. Keep alphanumeric tokens only ---
    tokens = [t for t in tokens if t.isalnum()]

    # --- 4. Stopword removal ---
    sw = set(stopwords.words('english'))
    clean_tokens = [t for t in tokens if t not in sw]

    # --- 5. Type–Token Ratio ---
    token_count = len(clean_tokens)
    type_count = len(set(clean_tokens))
    ttr = type_count / token_count if token_count > 0 else 0

    return {
        "tokens": clean_tokens,
        "token_count": token_count,
        "type_count": type_count,
        "ttr": ttr
    }

# Example
txt = "NLP models learn linguistic patterns more effectively when text is preprocessed."
print(preprocess_and_ttr_nltk(txt))
```

```Python
# Expt. 3: To study and implement regular expression
import re
text = "Price: 120 rupees, Discount: 25%"

#Find all words
words = re.findall(r"[A-Za-z]+", text)
print("Find all words: ", words)

#Extract numbers
nums = re.findall(r"\d+", text)
print("Extract numbers: ", nums)

#Remove punctuation
clean = re.sub(r"[^A-Za-z0-9\s]", " ", text)
print("Removing Punctuation: ", clean)

#Split on Punctuation
sentences = re.split(r"[.!?:%]", text)
print("Split at punct: ", sentences)

# 1. Validate an email
import re

email = "test_user123@mail.com"
pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

if re.match(pattern, email):
    print("Valid")
else:
    print("Invalid")

2. Find all numbers
numbers = re.findall(r'\d+', text)
print(f"Numbers found: {numbers}")
indian_numbers = re.findall(r'\+91[-\s]?\d{10}', text)
print(f"Indian Numbers (+91): {indian_numbers}")

# 3. Replace all numbers with '#'
text_with_numbers_masked = re.sub(r'\d+', '#', text)
print(f"Text after replacing numbers:\n{text_with_numbers_masked}")

# 4. Check if the word 'call' appears (case-insensitive)
pattern = r'call'
match = re.search(pattern, text, re.IGNORECASE)
if match:
 print(f"The word 'call' was found at position {match.start()}.")
else:
 print("The word 'call' was not found.")

# 5. Find repeated characters
import re

text = "loooool what is happening"
repeats = re.findall(r"(.)\1+", text)
print(repeats)
```

```Python
# Expt. 4: To study and implement n-grams probability
from collections import defaultdict

# SAMPLE TEXT
text = "I love coffee and I love cakes"
words = text.split()


# FUNCTION: make n-grams

def get_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

# n-grams
unigrams = get_ngrams(words, 1)
bigrams = get_ngrams(words, 2)
trigrams = get_ngrams(words, 3)


# FREQUENCY COUNTS

def freq(ngrams):
    f = defaultdict(int)
    for n in ngrams:
        f[n] += 1
    return f

uni_f = freq(unigrams)
bi_f = freq(bigrams)
tri_f = freq(trigrams)

---
# TOTAL COUNTS

total_uni = sum(uni_f.values())
total_bi = sum(bi_f.values())
total_tri = sum(tri_f.values())

print("Unigram freq:", dict(uni_f))
print("Total:", total_uni)
print("Bigram freq:", dict(bi_f))
print("Total:", total_bi)
print("Trigram freq:", dict(tri_f))
print("Total:", total_tri)

# UNIGRAM PROBABILITIES

print("\n--- UNIGRAM PROBABILITIES ---")
uni_prob = {}
for (w,), c in uni_f.items():
    uni_prob[w] = c / total_uni
    print(f"P({w}) = {uni_prob[w]:.4f}")


# BIGRAM CONDITIONAL PROBABILITIES

print("\n--- BIGRAM PROBABILITIES P(w2 | w1) ---")
bi_prob = {}
for (w1, w2), c in bi_f.items():
    bi_prob[(w1, w2)] = c / uni_f[(w1,)]
    print(f"P({w2} | {w1}) = {bi_prob[(w1, w2)]:.4f}")


# TRIGRAM CONDITIONAL PROBABILITIES

print("\n--- TRIGRAM PROBABILITIES P(w3 | w1 w2) ---")
tri_prob = {}
for (w1, w2, w3), c in tri_f.items():
    tri_prob[(w1, w2, w3)] = c / bi_f[(w1, w2)]
    print(f"P({w3} | {w1} {w2}) = {tri_prob[(w1, w2, w3)]:.4f}")
```

```Python
# Expt. 5: To study and implement segmentation, tokenization, stemming,
# legalization, lemmatization, and parts of speech tagging
import nltk
from nltk.stem import PorterStemmer
import spacy

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Sample Text
text = "The children are playing in the gardens and studies are being conducted."

#Tokenize
words = nltk.word_tokenize(text)
print("Tokens:", words)

#Stemming
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in words]
print("Stemming:", stems)

#Lemmatization using SpaCy lemmatization library
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

lemmas = [token.lemma_ for token in doc]
print("Lemmatization:", lemmas)

#POS Tagging
pos_tags = nltk.pos_tag(words)
print("POS Tags:", pos_tags)

#POS Tagging + Lemmatization
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print('\nspaCy POS + Lemmas:')
for token in doc:
  print(f"{token.text:12} | {token.pos_:6} | {token.lemma_}")
```

```Python
# Expt. 6: To study and implement hidden markov model

import numpy as np
# 1. Define states
states = ['Hot', 'Cold']
# 2. Define observations
observations = ['Walk', 'Shop', 'Clean']
# 3. Define initial state probabilities
start_probability = {'Hot': 0.8, 'Cold': 0.2}
# 4. Define transition probabilities
transition_probability = {
 'Hot': {'Hot': 0.7, 'Cold': 0.3},
 'Cold': {'Hot': 0.4, 'Cold': 0.6}
}
# 5. Define emission probabilities
emission_probability = {
 'Hot': {'Walk': 0.2, 'Shop': 0.4, 'Clean': 0.4},
 'Cold': {'Walk': 0.6, 'Shop': 0.3, 'Clean': 0.1}
 }

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for st in states:
        V[0][st] = start_p[st] * emit_p[st].get(obs[0], 0)
        path[st] = [st]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for st in states:
            (prob, state) = max(
                (V[t - 1][prev_st] * trans_p[prev_st].get(st, 0) * emit_p[st].get(obs[t], 0), prev_st)
                for prev_st in states
            )
            V[t][st] = prob
            newpath[st] = path[state] + [st]

        path = newpath

    # Find the maximum probability and the corresponding path
    (prob, state) = max((V[len(obs) - 1][st], st) for st in states)
    return prob, path[state]

print("States:", states)
print("Observations:", observations)
print("Initial Probabilities:", start_probability)
print("Transition Probabilities:", transition_probability)
print("Emission Probabilities:", emission_probability)

example_obs = ['Walk', 'Shop', 'Clean']
probability, path = viterbi(example_obs, states, start_probability,
transition_probability, emission_probability)
print("Most likely state sequence for observation sequence:",
example_obs)
print("Sequence:", path)
print("Probability:", probability)
```

```Python
# Expt. 7: To study and implement Name entity recognition(NER)
!pip install spacy
!python -m spacy download en_core_web_sm
#restart session

import spacy

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Sample text for NER
text = """
Apple is looking at buying U.K. startup for $1 billion.
Elon Musk visited India in 2024 to discuss Tesla’s entry plans.
Mumbai is experiencing rapid growth in the IT sector.
"""

# Process the text
doc = nlp(text)

print("\n=== Named Entity Recognition (NER) Results ===")
for ent in doc.ents:
    print(f"Entity: {ent.text:25}  | Label: {ent.label_}")
    
print("\nTotal Named Entities Detected:", len(doc.ents))
```

```Python
# Expt. 8: To study and implement wordnet in NLP

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

# Example Word: "great"
word = "bank"
print(f"Exploring WordNet for the word: '{word}'")

# 1. Get Synsets (Sets of synonyms)
word = "bank"
synsets = wn.synsets(word)
print("Synsets:", synsets)

# 2. Access Lemmas (Words in the synset)
for s in synsets:
    print("\nSynset:", s.name())
    print("Lemmas:", [lemma.name() for lemma in s.lemmas()])

# 3. Definitions & Examples
for s in synsets:
    print("\nSynset:", s.name())
    print(">> Definition:", s.definition())
    print(">> Examples:", s.examples())

# 4. Hypernyms (More general concepts)
syn = wn.synsets("dog")[0]   # choose the most common sense
print("Hypernyms:", syn.hypernyms())

# 5. Hyponyms (More specific concepts)
print("Hyponyms:", syn.hyponyms())

# 6. Part-of-speech (POS) filtering
nouns = wn.synsets("run", pos=wn.NOUN)
verbs = wn.synsets("run", pos=wn.VERB)

print("Noun senses:", nouns)
print("Verb senses:", verbs)

# 7. Word Similarity (Path similarity)
dog = wn.synsets("dog")[0]
cat = wn.synsets("cat")[0]

print("Path similarity:", dog.path_similarity(cat))
print("WUP similarity:", dog.wup_similarity(cat))

# 8. Antonyms
happy = wn.synsets("happy")[0]

for lemma in happy.lemmas():
    if lemma.antonyms():
        print("Antonym of happy:", lemma.antonyms()[0].name())

9. Get all words in WordNet for a POS
all_nouns = list(wn.all_synsets(pos=wn.NOUN))
print("Total noun synsets:", len(all_nouns))
```

```Python
# Expt. 9:  To study and implement Probabilistic context free grammar (PCFG) in NLP

# pip install nltk
from nltk.grammar import PCFG
from nltk.parse import ViterbiParser, ChartParser
from nltk import Tree
from nltk.grammar import Nonterminal
# --- Define a PCFG with explicit probabilities ---
grammar_str = """
S -> NP VP [1.0]
# NP alternatives (sums to 1.0)
NP -> Det N [0.4]
NP -> NP PP [0.3]
NP -> Pronoun [0.3]
# VP alternatives (sums to 1.0)
VP -> V NP [0.5]
VP -> VP PP [0.5]
PP -> P NP [1.0]
Det -> 'the' [0.9]
Det -> 'a' [0.1]
N -> 'man' [0.6]
N -> 'telescope' [0.4]
Pronoun -> 'I' [1.0]
V -> 'saw' [1.0]
P -> 'with' [1.0]
"""
pcfg = PCFG.fromstring(grammar_str)
print("Loaded PCFG:\n")
print(pcfg)
print("\n" + "="*60 + "\n")
# --- Parsers ---
viterbi = ViterbiParser(pcfg)
chart = ChartParser(pcfg)
# Ambiguous sentence
sentence = "I saw the man with a telescope"
tokens = sentence.split()
# Helper: compute probability of a Tree under the PCFG by multiplying
# production probabilities
def tree_probability(tree: Tree, grammar: PCFG) -> float:
    prod_probs = {prod: prod.prob() for prod in grammar.productions()}
    prob = 1.0
    # recursively walk the tree and collect productions
    def walk(t):
        nonlocal prob
        if isinstance(t, Tree):
            lhs = Nonterminal(t.label())
            rhs = []
            for child in t:
                if isinstance(child, Tree):
                    rhs.append(Nonterminal(child.label()))
                else:
                    rhs.append(child)
            # find matching production in grammar (there should be
            # exactly one match for PCFG grammar rules)
            matched = None
            for p in grammar.productions(lhs=lhs):
                # compare RHS symbols as strings/Nonterminals
                # convert p.rhs() to comparable form
                rhs_form = []
                for x in p.rhs():
                    if isinstance(x, Nonterminal):
                        rhs_form.append(Nonterminal(str(x)))
                    else:
                        rhs_form.append(x)
                if tuple(rhs_form) == tuple(rhs):
                    matched = p
                    break
            if matched is None:
                # if no production match found, the probability is 0
                # (shouldn't happen if tree is from this grammar)
                return 0.0
            prob *= matched.prob()
            # recurse
            for child in t:
                if isinstance(child, Tree):
                    walk(child)
    walk(tree)
    return prob
# --- Get most probable parse (Viterbi) ---
print("Most probable parse (Viterbi):")
viterbi_trees = list(viterbi.parse(tokens))
if viterbi_trees:
    best = viterbi_trees[0]
    print(best)
    try:
        best.pretty_print()
    except Exception:
        pass
    print(f"Probability (Viterbi tree .prob if available): "
          f"{getattr(best, 'prob', lambda: tree_probability(best, pcfg))()}\n")
else:
    print("No Viterbi parse found.\n")
# --- Get all parses (ChartParser) to show ambiguity ---
print("="*40)
print("All parses (ChartParser) - demonstrates ambiguity:")
all_trees = list(chart.parse(tokens))
if not all_trees:
    print("No parses found with this grammar.")
else:
    for i, t in enumerate(all_trees, start=1):
        print(f"\nParse #{i}:")
        print(t)
        try:
            t.pretty_print()
        except Exception:
            pass
        # compute probability under PCFG
        prob = tree_probability(t, pcfg)
        print(f"Computed probability (product of rule probs): "
              f"{prob:.8f}")
# --- Compare probabilities and indicate which parse is preferred ---
if len(all_trees) > 1:
    probs = [(i+1, tree_probability(t, pcfg), t) for i,t in
             enumerate(all_trees)]
    probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)
    print("\n" + "="*40)
    print("Parses sorted by probability (highest first):")
    for idx, p, t in probs_sorted:
        print(f"Parse #{idx} \u2014 probability = {p:.8f}")
    best_idx, best_prob, best_tree = probs_sorted[0]
    print(f"\nViterbi selected parse #{best_idx} with probability "
          f"{best_prob:.8f}.")
```

```Python
# E10: INFORMATION RETRIEVAL USING TF-IDF

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# 1. Example Document Collection
# -----------------------------
documents = [
    "The sky is blue and beautiful",
    "Love this blue and bright sky",
    "The quick brown fox jumps over the lazy dog",
    "A king's breakfast includes cereal and milk",
    "The sky is clear and the stars are twinkling tonight"
]

# -----------------------------
# 2. Preprocessing
# -----------------------------
stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

clean_docs = [preprocess(doc) for doc in documents]

# -----------------------------
# 3. Vectorize using TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(clean_docs)

# -----------------------------
# 4. Query Input
# -----------------------------
query = "blue sky tonight"
query_clean = preprocess(query)
query_vec = vectorizer.transform([query_clean])

# -----------------------------
# 5. Cosine Similarity
# -----------------------------
similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

# -----------------------------
# 6. Rank Documents
# -----------------------------
ranked_indices = np.argsort(similarity_scores)[::-1]

print("\nQuery:", query)
print("\nTop Relevant Documents:\n")
for i in ranked_indices:
    print(f"Score: {similarity_scores[i]:.4f}  -->  {documents[i]}")

```
