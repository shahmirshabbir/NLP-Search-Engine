import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import faiss
from itertools import product
from sentence_transformers import SentenceTransformer

class NLP:
    def __init__(self, directory):
        self.modelPath = directory + '/all-MiniLM-L6-v2'
        self.indexPath = directory + '/faiss_index.index'
        self.mappingsPath = directory + '/product_mapping.pkl'
        self.dataPath = directory + '/my_data.csv'

        self.model = self.getBertModel()
        self.product_mapping = None
        self.index = None
        self.train_data = None

        if self.isTrained():
          print('pre trained')
          self.index = self.getFaissIndex()
          self.train_data = self.getCsvFile()
          self.product_mapping = self.getMappings()
        else :
          print('training')
          self.train()

    def getMappings(self):
      mapping = joblib.load(self.mappingsPath)
      return mapping

    def saveMappings(self,mappings):
      joblib.dump(mappings, self.mappingsPath)

    def updateMappings(self,products):
      new_mapping = {product: idx for idx, product in enumerate(products["ProductID"].unique() , start = len(self.product_mapping.values()))}
      self.product_mapping.update(new_mapping)
      self.saveMappings(self.product_mapping)

    def getCsvFile(self):
      df = pd.read_csv(self.dataPath)
      return df

    def saveCsvFile(self,df):
      df.to_csv(self.dataPath, index=False)

    def updateData(self,df):
      data = self.getCsvFile()
      df1 = pd.concat([data, df] ,ignore_index=True)
      self.train_data = df1
      self.saveCsvFile(df1)

    def cleanCsvFile(self,df):
      df = df[df['Status'] != 'inactive']
      df = df.reset_index(drop=True)
      return df

    def getBertModel(self):
      return SentenceTransformer(self.modelPath)

    def getFaissIndex(self):
      index = faiss.read_index(self.indexPath)
      return index

    def saveFaissIndex(self):
      faiss.write_index(self.index, self.indexPath)

    def updateFaissIndex(self ,vectors):
      combined_embeddings = np.vstack(vectors)

      self.index.add(combined_embeddings)
      faiss.write_index(self.index, self.indexPath)

    def isTrained(self):
      import os
      return os.path.isfile(self.indexPath)

    def train(self):

      self.train_data = self.getCsvFile()
      self.train_data = self.cleanCsvFile(self.train_data)
      self.train_data['combined_text'] = self.train_data.apply(self.combine_attributes, axis=1)
      self.train_data['clean'] = self.train_data['combined_text'].apply(self.preprocess)
      self.saveCsvFile(self.train_data)

      cleaned_products = [prod for prod in self.train_data['clean']]
      # print(self.train_data['combined_text'][0:10])

      self.product_vectors = self.model.encode(cleaned_products, convert_to_tensor=True)
      self.product_vectors = np.vstack(self.product_vectors)

      combined_embeddings = np.vstack(self.product_vectors)
      d = combined_embeddings.shape[1]

      print(self.product_vectors.shape)
      print(combined_embeddings.shape)
      # print(d)

      self.index = faiss.IndexFlatL2(d)
      self.index.add(combined_embeddings)
      self.saveFaissIndex()
      self.product_mapping = {product: idx for idx, product in enumerate(self.train_data["ProductID"].unique())}
      self.saveMappings(self.product_mapping)


    def preprocess(self,text):
      lemmatizer = WordNetLemmatizer()
      stop_words = set(stopwords.words('english'))
      text = text.lower()
      text = re.sub(r'[^a-zA-Z\s]', '', text)
      words = text.split()
      words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
      return ' '.join(words)

    def combine_attributes(self,row):
      attributes = [
          row['ProductName'],
          str(row['Price (INR)']),
          row['Gender'],
          row['Description'],
      ]
      combined_text = ' '.join(str(attr) for attr in attributes if attr)
      return ' '.join(set(combined_text.split()))

    def search(self,query, k=10 ):
      query_vector = self.model.encode(query, convert_to_tensor=True)
      query_vector = np.vstack(query_vector)
      query_vector = query_vector.reshape(1,384)

      indices = self.index.search(query_vector, k+1)[1]
      # print(indices)
      # indices = indices[1]
      result = [ self.train_data['clean'][i] for i in indices[0] if self.train_data['Status'][i] != 'inactive']
      c = 0
      while len(result) < k:
        c=c+1
        indices = self.index.search(query_vector, k+1+(k*c-len(result)))[1]
        result = [self.train_data['ProductID'][i] for i in indices[0] if self.train_data['Status'][i] != 'inactive']

      return result

    def delete(self,id):
      df = self.getCsvFile()
      df.loc[self.product_mapping[id], "Status"] = "inactive"
      self.saveCsvFile(df)
      self.train_data.loc[self.product_mapping[id], "Status"] = "inactive"

    def addProducts(self,data):
      data['combined_text'] = data.apply(self.combine_attributes, axis=1)
      cleaned_products = [self.preprocess(prod) for prod in data['combined_text']]
      tokenized_products = [word_tokenize(prod) for prod in cleaned_products]

      vectors = [self.get_weighted_vector(prod) for prod in cleaned_products]
      self.updateFaissIndex(vectors)
      self.updateData(data)
      self.updateMappings(data)
    def checkText(self):
      print(self.train_data['clean'][0:10])