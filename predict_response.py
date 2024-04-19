# Bibliotecas de pré-processamento de dados de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# palavras a serem ignoradas/omitidas ao enquadrar o conjunto de dados
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Biblioteca load_model
import tensorflow
from data_preprocessing import get_stem_words

# carregue o modelo
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Carregue os arquivos de dados
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bag_of_words = []

    # tokenize a entrada do usuário

    # converta a entrada do usuário em sua palavra-raiz: stemização

    # Remova duplicidades e classifique a entrada do usuário
   
    # Codificação de dados de entrada: crie a BOW para  user_input
    
    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # extraia a classe de predicted_class_label
   predicted_class = ""

   # agora que temos a tag prevista, selecione uma resposta aleatória

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # selecione uma resposta aleatória do robô
        bot_response = ""
    
        return bot_response
    

print("Oi, eu sou a Estela, como posso ajudar?")

while True:

    # obtenha a entrada do usuário
    user_input = input('Digite sua mensagem aqui: ')

    response = bot_response(user_input)
    print("Resposta do Robô: ", response)
