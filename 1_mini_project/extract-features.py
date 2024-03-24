#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download stopwords corpus if not already downloaded
nltk.download('stopwords')


external = {}
with open("resources/HSDB.txt") as h :
   for x in h.readlines() :
      external[x.strip().lower()] = "drug"
with open("resources/DrugBank.txt") as h :
   for x in h.readlines() :
      (n,t) = x.strip().lower().split("|")
      external[n] = t


   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"

def get_lemma(word):
    """
    Get the lemma of a word using NLTK's WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    return lemma

def is_stopword(word):
    """
    Check if a word is a stopword using NLTK's stopwords corpus.
    """
    stop_words = set(stopwords.words('english'))
    return word.lower() in stop_words
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   # single token - (token, start position in sentence, end position in sentence)
   result = []
   for k in range(0,len(tokens)):
      tokenFeatures = [];
      t = tokens[k][0] # actual token

      tokenFeatures.append("form="+t) # actual token
      tokenFeatures.append("suf3="+t[-3:]) # 3suffix of our token

      ###################################
      #### NEW FEATUES - OURS ######
      ###################################

      ####### original word but lower case
      tokenFeatures.append("form_lowercase="+t.lower())
      ####### lemma of the word
      tokenFeatures.append("lemma="+get_lemma(t)) 
      ####### if all capitalized
      if t.isupper(): tokenFeatures.append("all_capitalized="+str(True))
      ####### check if camelcase
      if bool(re.match(r'^[a-z]+(?:[A-Z][a-z]*)*$', t)): tokenFeatures.append("camelcase="+str(True))
      ####### presence of numbers
      if bool(re.search(r'\d', t)): tokenFeatures.append("digits_presence="+str(True))
      ####### presence of dashes
      if bool(re.search(r'[-–—]', t)): tokenFeatures.append("dash_presence="+str(True))
      ####### presence of +
      if '+' in t: tokenFeatures.append("plus_presence="+str(True))
      ####### presence of /
      if '/' in t: tokenFeatures.append("slash_presence="+str(True))
      ####### presence of |
      if '|' in t: tokenFeatures.append("pipe_presence="+str(True))
      ####### if starts from capitalized letter
      if t.istitle(): tokenFeatures.append("start_capital="+str(True))
      ###### checks if token is a stopword
      if is_stopword(t): tokenFeatures.append("stopword="+str(True))
      ####### has any punctuation
      if bool(re.search(r'[^\w\s]', t)): tokenFeatures.append("punctuation="+str(True))
      ###### existence of the token in external resources - folder resources
      # 4 different classes for different labels
      if t.lower() in external:
         tokenFeatures.append("resources="+str(external[t.lower()]))


      if k>0 :
         tPrev = tokens[k-1][0] # info about previous token
         tokenFeatures.append("formPrev="+tPrev) # actual previous token
         tokenFeatures.append("suf3Prev="+tPrev[-3:]) # 3suffix of our token

         ###################################
         #### NEW FEATUES - OURS ######
         ###################################

         ####### original word but lower case
         tokenFeatures.append("form_lowercasePrev="+tPrev.lower())
         ####### lemma of the word
         tokenFeatures.append("lemmaPrev="+get_lemma(tPrev)) 
         ####### if all capitalized
         if tPrev.isupper(): tokenFeatures.append("all_capitalizedPrev="+str(True))
         ####### check if camelcase
         if bool(re.match(r'^[a-z]+(?:[A-Z][a-z]*)*$', tPrev)): tokenFeatures.append("camelcasePrev="+str(True))
         ####### presence of numbers
         if bool(re.search(r'\d', tPrev)): tokenFeatures.append("digits_presencePrev="+str(True))
         ####### presence of dashes
         if bool(re.search(r'[-–—]', tPrev)): tokenFeatures.append("dash_presencePrev="+str(True))
         ####### presence of +
         if '+' in tPrev: tokenFeatures.append("plus_presencePrev="+str(True))
         ####### presence of /
         if '/' in tPrev: tokenFeatures.append("slash_presencePrev="+str(True))
         ####### presence of |
         if '|' in tPrev: tokenFeatures.append("pipe_presencePrev="+str(True))
         ###### if starts from capitalized letter
         if tPrev.istitle(): tokenFeatures.append("start_capitalPrev="+str(True))
         ###### check if token is a stopword
         if is_stopword(tPrev): tokenFeatures.append("stopwordPrev="+str(True))
         ####### has any punctuation
         if bool(re.search(r'[^\w\s]', tPrev)): tokenFeatures.append("punctuationPrev="+str(True))
         ##### existence of the token in external resources - folder resources
         # 4 different classes for different labels
         if tPrev.lower() in external:
            tokenFeatures.append("resourcesPrev="+str(external[tPrev.lower()]))

      else :
         tokenFeatures.append("BoS") # it does not have previous token

      if k<len(tokens)-1 : 
         tNext = tokens[k+1][0] # info about next token
         tokenFeatures.append("formNext="+tNext) # actual next token
         tokenFeatures.append("suf3Next="+tNext[-3:]) # 3suffix of our token

         ###################################
         #### NEW FEATUES - OURS ######
         ###################################

         ####### original word but lower case
         tokenFeatures.append("form_lowercaseNext="+tNext.lower())
         ####### lemma of the word
         tokenFeatures.append("lemmaNext="+get_lemma(tNext))
         ####### if all caps
         if tNext.isupper(): tokenFeatures.append("all_capitalizedNext="+str(True))
         ####### check if camelcase
         if bool(re.match(r'^[a-z]+(?:[A-Z][a-z]*)*$', tNext)): tokenFeatures.append("camelcaseNext="+str(True))
         ####### presence of numbers
         if bool(re.search(r'\d', tNext)): tokenFeatures.append("digits_presenceNext="+str(True))
         ####### presence of dashes
         if bool(re.search(r'[-–—]', tNext)): tokenFeatures.append("dash_presenceNext="+str(True))
         ####### presence of +
         if '+' in tNext: tokenFeatures.append("plus_presenceNext="+str(True))
         ####### presence of /
         if '/' in tNext: tokenFeatures.append("slash_presenceNext="+str(True))
         ####### presence of |
         if '|' in tNext: tokenFeatures.append("pipe_presenceNext="+str(True))
         ####### if starts from capitalized letter
         if tNext.istitle(): tokenFeatures.append("start_capitalNext="+str(True))
         ###### checks if token is a stopword
         if is_stopword(tNext): tokenFeatures.append("stopwordNext="+str(True))
         ####### has any punctuation
         if bool(re.search(r'[^\w\s]', tNext)): tokenFeatures.append("punctuationNext="+str(True))
         ##### existence of the token in external resources - folder resources
         # 4 different classes for different labels
         if tNext.lower() in external:
            tokenFeatures.append("resourcesNext="+str(external[tNext.lower()]))
      else:
         tokenFeatures.append("EoS") # it does not have next token

      result.append(tokenFeatures)


   return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)):
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans)
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
