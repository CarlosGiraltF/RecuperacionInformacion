#Metodos de carga y procesamiento de datos usados en el resto de ejercicios de la práctica
import pandas as pd, re, numpy as np, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
import matplotlib.pyplot as plt

# Método para leer los ficheros tabulares del ejercicio de clasificación de texto (clasificación, título y descripción)
# Lee un fichero en un dataframe de Pandas y junta el título con la descripción
# Selecciona aleatoriamente un 10% de los datos. Esto esta hecho en el ejercicio para que el entrenamiento sea más rápido a costa de precisión.
def __leeDataFrameClasificador(file):
    df = pd.read_csv(file, index_col=False)
    df['Text'] = df['Title'] + '. ' + df['Description']
    df.drop(['Title', 'Description'], axis=1, inplace=True)
    df_muestra = df.sample(frac=0.1, random_state=0)
    return df_muestra

# Procesa una cadena de texto para eliminar simbolos de puntuación y otros caracteres no alfanumericos y acentos.
# Convierte el texto a minuscula y elimina espacios extra.
def __limpiaCadenasDeTexto(docs):
  norm_docs = []
  for doc in docs:
    doc = re.sub(r'[^a-zA-Z0-9\s\n\t\r]', ' ', doc).lower()
    doc = re.sub(' +', ' ', doc).strip()
    norm_docs.append(doc)
  return norm_docs

# tokeniza el texto y lo conviete en vectores de longitud constante aññadiendo tokens comodin para frases cortas
# el código describe como ajustar el tamaño de los vectores generados a la cadena mas larga, pero el tamaño se ha limitado
# a 200 para que el entrenamiento vaya más rapido a costa de la precisión
def __tokenizadorTexto(X_entren, X_test):
    t = Tokenizer(oov_token='<UNK>')
    t.fit_on_texts(X_entren)
    t.word_index['<PAD>'] = 0
    max_num_columns = np.max([len(row) for row in X_entren] + [len(row) for row in X_test])
    max_num_columns = 200
    X_entrenT = pad_sequences(t.texts_to_sequences(X_entren), maxlen=max_num_columns)
    X_testT = pad_sequences(t.texts_to_sequences(X_test), maxlen=max_num_columns)
    return X_entrenT, X_testT, len(t.word_index)

# devuelve los datos de entrenamiento y test del clasificador
# lee los datos, los limpia, y tokeniza. Las categorias las convierte a one-hot.
def lecturaDatosEntrenamientoYTestClasificador():
    dataset_entrenamiento = __leeDataFrameClasificador('datos/clasificacionEntrenamiento.csv')
    dataset_test = __leeDataFrameClasificador('datos/clasificacionTest.csv')
    X_entren = __limpiaCadenasDeTexto(dataset_entrenamiento['Text'].values)
    X_test = __limpiaCadenasDeTexto(dataset_test['Text'].values)
    X_entren, X_test, tamVoc = __tokenizadorTexto(X_entren, X_test)
    y_entren = to_categorical(dataset_entrenamiento['Class Index'].values - 1)
    y_test = to_categorical(dataset_test['Class Index'].values - 1)
    return (X_entren, y_entren, X_test, y_test, tamVoc)

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
    df = pd.read_csv('datos/clasificacionEntrenamiento.csv', index_col=False)
    print('---------------------------------------------------')
    print('Estructura de los datos')
    print(df.head(5));  print(df.tail(5))

    #El proceso de limpieza deja solo token utiles para el proceso de aprendizaje.
    datosLimpios = __limpiaCadenasDeTexto((df['Description'].values))
    print('\n---------------------------------------------------')
    print('Ejemplo de limpieza de datos (antes/despues)')
    print('---------------------------------------------------')
    print(df['Description'][0])
    print(datosLimpios[0])

    #El proceso de tokenización crea un vector de palabras ajustado para que tengan todos el mismo tamaño
    t = Tokenizer(oov_token='<UNK>')
    t.fit_on_texts(datosLimpios)
    t.word_index['<PAD>'] = 0
    print('\n---------------------------------------------------')
    print('Ejemplo de tokenización de la frase anterior')
    print('---------------------------------------------------')
    print(datosLimpios[0])
    tokens = t.texts_to_sequences([datosLimpios[0]])
    print(tokens)
    print(pad_sequences(tokens, maxlen=20))
    print(list(t.word_index.items())[:10])

    #respecto a las clasificaciones, simplemente las transformamos en representacion one-hot
    y_entren = to_categorical(df['Class Index'].values - 1)
    print('\n---------------------------------------------------')
    print('Ejemplo de conversión de las categorias')
    print('---------------------------------------------------')
    print (df['Class Index'][0], y_entren[0])


