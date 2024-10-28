#Metodos de carga y procesamiento de datos usados en el resto de ejercicios de la práctica
import pandas as pd, re, numpy as np, os, unicodedata

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
import matplotlib.pyplot as plt

# Método para leer el fichero tabular de traduccion Ingles a Español. Se le quita la columna licencia
# Lee un fichero en un dataframe de Pandas y junta el título con la descripción
# Selecciona aleatoriamente un 10% de los datos. Esto esta hecho en el ejercicio para que el entrenamiento sea más rápido a costa de precisión.
def __leeDataFrameTraductor(file):
    df = pd.read_csv(file, index_col=False, delimiter='\t', names=['Ingles', 'Espanol', 'Licencia'])
    df.drop(['Licencia'], axis=1, inplace=True)
    df_muestra = df.sample(frac=0.3, random_state=0)
    return df_muestra

# Procesa una cadena de texto para eliminar simbolos de puntuación y otros caracteres no alfanumericos y acentos.
# Convierte el texto a minuscula y elimina espacios extra.
def __limpiaCadenasDeTexto(docs):
  norm_docs = []
  for doc in docs:
    doc = ''.join(c for c in unicodedata.normalize('NFD', doc) if unicodedata.category(c) != 'Mn')
    doc = re.sub(r'[^a-zA-Z0-9\s\n\t\r]', ' ', doc).lower()
    doc = re.sub(' +', ' ', doc).strip()
    norm_docs.append(doc)
  return norm_docs

# tokeniza el texto y lo conviete en vectores de longitud constante aññadiendo tokens comodin para frases cortas
# el código describe como ajustar el tamaño de los vectores generados a la cadena mas larga, pero el tamaño se ha limitado
# a 10 para que el entrenamiento vaya más rapido a costa de la precisión
def __tokenizadorTexto(datos):
    t = Tokenizer(oov_token='<UNK>')
    t.fit_on_texts(datos)
    t.word_index['<PAD>'] = 0
    #maxlen = int(np.max([len(row) for row in datos]))
    maxlen = 10
    datosT = pad_sequences(t.texts_to_sequences(datos), maxlen=maxlen, padding='post')
    return datosT, t, maxlen

# devuelve los datos de entrenamiento y test del clasificador
# lee los datos, los limpia, y tokeniza. Lo hace tanto para los textos en Ingles como en Español
def lecturaDatosEntrenamientoYTestTraductor():
    dataset = __leeDataFrameTraductor('datos/traductorFrasesEnEs.txt')
    entren = __limpiaCadenasDeTexto(dataset['Ingles'].values)
    test = __limpiaCadenasDeTexto(dataset['Espanol'].values)
    X_T, X_tokenizer, X_maxlen = __tokenizadorTexto(entren)
    y_T, y_tokenizer, y_maxlen = __tokenizadorTexto(test)
    indice_div = int(len(X_T) * 0.8)
    X_entren = X_T[:indice_div]
    y_entren = y_T[:indice_div]
    X_test = X_T[indice_div:]
    y_test = y_T[indice_div:]
    return X_entren, y_entren, X_test, y_test, X_tokenizer, y_tokenizer, X_maxlen, y_maxlen

#Método para visualizar una serie de datos con las etiquetas indicadas en los ejes
def visualizaSerieDatos(datos,etiquetaX, etiquetaY):
    plt.figure(figsize=(10, 5))
    plt.plot(datos)
    plt.xlabel(etiquetaX, fontsize=15)
    plt.ylabel(etiquetaY, fontsize=15)
    plt.show()

#Código de ejemplo para estudiar los datos y estructuras generadas por cada método
if __name__ == '__main__':
    #La estructura de los datos es un fichero tabular con titulo, texto y categoría
    #La descripción a veces incluye la agencia de noticias y a veces no.
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('datos/traductorFrasesEnEs.txt', index_col=False, delimiter='\t', names=['Ingles', 'Espanol', 'Licencia'])
    print('---------------------------------------------------')
    print('Estructura de los datos')
    print(df.head(5)); print(df.tail(5))

    #El proceso de limpieza deja solo token utiles para el proceso de aprendizaje.
    entren = __limpiaCadenasDeTexto(df['Ingles'].values)
    test = __limpiaCadenasDeTexto(df['Espanol'].values)
    print('\n---------------------------------------------------')
    print('Ejemplo de limpieza de datos (antes/despues)')
    print('---------------------------------------------------')
    print(df['Ingles'][10000], df['Espanol'][10000])
    print(entren[10000], test[10000])

    #El proceso de tokenización crea un vector de palabras ajustado para que tengan todos el mismo tamaño
    t = Tokenizer(oov_token='<UNK>')
    t.fit_on_texts(entren)
    t.word_index['<PAD>'] = 0
    t2 = Tokenizer(oov_token='<UNK>')
    t2.fit_on_texts(test)
    t2.word_index['<PAD>'] = 0
    print('\n---------------------------------------------------')
    print('Ejemplo de tokenización de las frases anteriores')
    print('---------------------------------------------------')
    print(entren[10000])
    tokens = t.texts_to_sequences([entren[10000]])
    print(tokens)
    print(pad_sequences(tokens, maxlen=10))
    print(list(t.word_index.items())[:10])
    print(test[10000])
    tokens = t2.texts_to_sequences([test[10000]])
    print(tokens)
    print(pad_sequences(tokens, maxlen=10))
    print(list(t2.word_index.items())[:10])
