#Metodos de carga y procesamiento de datos usados en el resto de ejercicios de la práctica
import unicodedata
import pandas as pd, re, numpy as np, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from keras.utils import set_random_seed
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling1D
from keras_nlp.layers import TransformerEncoder, TokenAndPositionEmbedding
import matplotlib.pyplot as plt
import sys
import xml.etree.ElementTree as ET

#################################################
#       PARTE 1:CREAR DATOS DE ENTRENAMIENTO
#################################################

ns= {'dc':'http://purl.org/dc/elements/1.1/'}   #definimos el espacio de nombres


# Define las palabras clave para cada categoría
salud_keywords = {
    'Medicina', 'Enfermería', 'Médico', 'Enfermero', 'Enfermera', 
    'Fisioterapia', 'Veterinaria', 'Terapia Ocupacional', 'Nutrición'
}

ingenieria_keywords = {
    'Ingeniero', 'Ingeniera', 'Ingeniería', 
    'Ingeniería de Tecnologías y Servicios de Telecomunicación',
    'Ingeniería Electrónica', 'Ingeniería Química', 'Ingeniería Mecánica', 
    'Ingeniería Civil', 'Ingeniería de Organización Industrial', 
    'Ingeniería en Diseño Industrial', 'Ingeniería Agroalimentaria', 
    'Ingeniería Informática', 'Ingeniería Mecatrónica'
}

magisterio_keywords = {
    'Magisterio', 'Educación', 'educación', 'Infantil', 
    'Educación Primaria', 'Maestro', 'Maestra'
}

humanidades_keywords = {
    'Estudios Ingleses', 'Filología', 'Historia del Arte', 
    'Historia', 'Filosofía', 'Geografía', 'Derecho'
}

economia_keywords = {
    'Economía', 'ADE', 'Administración', 'Dirección de Empresas', 
    'Finanzas', 'Marketing', 'Investigación de Mercados'
}

ciencias_keywords = {
    'Matemáticas', 'Física', 'Química', 'Geología', 'Óptica', 
    'Ciencias Ambientales', 'Ciencia y Tecnología de los Alimentos'
}

ciencias_sociales_keywords = {
    'Psicología', 'Trabajo Social', 'Relaciones Laborales'
}


otros_keywords = {
    'Periodismo', 'Actividad Física', 'Bellas Artes'
}

# Define las categorías y palabras clave
categorias = {
    'c0': salud_keywords,                         # Salud
    'c1': ingenieria_keywords,                    # Ingeniería
    'c2': magisterio_keywords,                    # Magisterio
    'c3': humanidades_keywords,                   # Humanidades
    'c4': economia_keywords,                       # Economía
    'c5': ciencias_keywords,                       # Ciencias
    'c6': ciencias_sociales_keywords,              # Ciencias Sociales
    'c7': otros_keywords                          # Otros
}

def limpiar_texto(texto):
    if texto is None:
        return ''

    # Normalizar texto a Unicode NFKD para manejar caracteres especiales
    texto = unicodedata.normalize('NFKD', texto)
    
    # Filtrar caracteres no ASCII o no imprimibles
    texto = ''.join(c for c in texto if unicodedata.category(c)[0] != 'C' and c.isprintable())
    
    # Reemplazar cualquier secuencia de caracteres que no sea alfanumérica o espacios por un espacio
    texto = re.sub(r'[^a-zA-Z0-9\s]', ' ', texto)
    
    # Remover múltiples espacios y hacer strip para quitar espacios al inicio y final
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

# Método para leer los ficheros tabulares del ejercicio de clasificación de texto (clasificación, título y descripción)
# Lee un fichero en un dataframe de Pandas y junta el título con la descripción
# Selecciona aleatoriamente un 10% de los datos. Esto esta hecho en el ejercicio para que el entrenamiento sea más rápido a costa de precisión.
def __leeDatos(file, filesSet):
    tree = ET.parse(file)
    root = tree.getroot()
    type = root.find('dc:type',ns)
    subject = root.find('dc:subject',ns)
    if subject is not None:
        subject_text=subject.text
    if type is not None and type.text == "TAZ-TFG" and "Graduado en" in subject_text:
        filesSet.add(subject_text)


# Clasifica los valores del conjunto files_set en las categorías definidas
def clasifica_files_set(files_set):
    clasificaciones = {key: set() for key in categorias}  # Diccionario para almacenar clasificaciones

    for item in files_set:
        for categoria, keywords in categorias.items():
            if any(keyword.lower() in item.lower() for keyword in keywords):  # Compara en minúsculas
                clasificaciones[categoria].add(item)
                break  # Sal del bucle al encontrar la primera coincidencia de categoría

    return clasificaciones

def determinar_categoria(subject):
    for categoria, keywords in categorias.items():
        for keyword in keywords:
            if keyword in subject:
                return categoria
    return 'c7'  # Categoría "Otros" si no se encuentra ninguna coincidencia

def escribe(file,OutputFile):
    tree = ET.parse(file)
    root = tree.getroot()
    type = root.find('dc:type',ns)
    subject = root.find('dc:subject',ns)
    if subject is not None:
        subject_text=subject.text
    if type is not None and type.text == "TAZ-TFG" and "Graduado en" in subject_text:
        title= root.find('dc:title',ns)
        description=root.find('dc:description',ns)
        if title is not None and description is not None:
                # Determinar la categoría
            categoria = determinar_categoria(subject_text)
            #with open(OutputFile, 'a', encoding='utf-8') as f:
            # Escribir en el archivo de salida
            OutputFile.write(limpiar_texto(title.text)+'\t;'+limpiar_texto(description.text)+'\t;'+categoria+'\n') 
        


def crearArchivoEntrenamiento(FolderName,OutputFile):
        # Crear conjunto de categorías encontradas
    files_set = set()

    # Recorrer todos los archivos en la carpeta especificada
    for file_name in os.listdir(FolderName):
        file_path = os.path.join(FolderName, file_name)
        if os.path.isfile(file_path):  # Verificar que es un archivo
            __leeDatos(file_path, files_set)

    clasificacion = clasifica_files_set(files_set)
    # Asegurarse de que el archivo de salida existe
    if not os.path.exists(OutputFile):
        with open(OutputFile, 'w', encoding='utf-8') as file:  # Crea el archivo si no existe
            file.write('Title\t;Description\t;Class Index\n')

    # Escribir los resultados en el archivo de salida
   # with open(OutputFile, 'a', encoding='utf-8') as file:  # Abre en modo de agregar
            for file_name in os.listdir(FolderName):
                file_path = os.path.join(FolderName, file_name)
                if os.path.isfile(file_path):  # Verificar que es un archivo
                    escribe(file_path,file)


#################################################
#       PARTE 2:PREPARAR DATOS DE ENTRENAMIENTO
#################################################
# Método para leer los ficheros tabulares del ejercicio de clasificación de texto (clasificación, título y descripción)
# Lee un fichero en un dataframe de Pandas y junta el título con la descripción
def __leeDataFrameClasificador(file):
    df = pd.read_csv(file, sep='\t;',engine='python',index_col=False,encoding='utf-8')
    df['Text'] = df['Title'] + '. ' + df['Description']
    df.drop(['Title', 'Description'], axis=1, inplace=True)
    return df

# Procesa una cadena de texto para eliminar simbolos de puntuación y otros caracteres no alfanumericos y acentos.
# Convierte el texto a minuscula y elimina espacios extra.
def __limpiaCadenasDeTexto(docs):
  norm_docs = []
  for doc in docs:
    doc = str(doc).lower()
    doc = re.sub(r'á', 'a', str(doc))
    doc = re.sub(r'é', 'e', str(doc))
    doc = re.sub(r'í', 'i', str(doc))
    doc = re.sub(r'ó', 'o', str(doc))
    doc = re.sub(r'ú', 'u', str(doc))
    doc = re.sub(r'ü', 'u', str(doc))
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
    #dataset_entrenamiento = __leeDataFrameClasificador('datos/clasificacionEntrenamiento.csv')
    dataset_entrenamiento = __leeDataFrameClasificador('datos/clasificacionEntrenamientoZaguan.csv')
    dataset_test = __leeDataFrameClasificador('datos/clasificacionTestZaguan.csv')
    X_entren = __limpiaCadenasDeTexto(dataset_entrenamiento['Text'].values)
    X_test = __limpiaCadenasDeTexto(dataset_test['Text'].values)
    X_entren, X_test, tamVoc = __tokenizadorTexto(X_entren, X_test)
    y_entren = to_categorical(dataset_entrenamiento['Class Index'].values - 1,num_classes=8)
    y_test = to_categorical(dataset_test['Class Index'].values - 1,num_classes=8)
    return (X_entren, y_entren, X_test, y_test, tamVoc)

#Método para visualizar una serie de datos con las etiquetas indicadas en los ejes
def visualizaSerieDatos(datos,etiquetaX, etiquetaY):
    plt.figure(figsize=(10, 5))
    plt.plot(datos)
    plt.xlabel(etiquetaX, fontsize=15)
    plt.ylabel(etiquetaY, fontsize=15)
    plt.show()

#################################################
#       PARTE 3:CREAR MODELO
#################################################

def createModel(tamVoc,tamFrase,tamEmbd):
    model = Sequential()
    model.add(TokenAndPositionEmbedding(tamVoc, tamFrase, tamEmbd))
    model.add(TransformerEncoder(32, num_heads=3 ))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='CategoricalCrossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    return model

if __name__ == '__main__':
    FolderName="../../recordsdc/"
    OutputFile="datos/clasificacionEntrenamientoZaguan.csv"
    # Procesa argumentos del sistema
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-docs' and i + 1 < len(sys.argv):
            FolderName = sys.argv[i + 1]

        if sys.argv[i] == '-output' and i + 1 < len(sys.argv):
            OutputFile = sys.argv[i + 1]

    crearArchivoEntrenamiento(FolderName,OutputFile)
    X_entren, y_entren, X_test, y_test, tamVoc = lecturaDatosEntrenamientoYTestClasificador()
    model=createModel(tamVoc,len(X_entren[0]),  50)
    history = model.fit(X_entren, y_entren, epochs=10, validation_steps=10, batch_size=64 , verbose=0)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Precisión del modelo con los test: %.2f%%" % (scores[1] * 100))
    print("Ejemplo de clasificación de la componente 0 del test")
    print("Categoria real: ", np.argmax(y_test[0]) + 1, " Categoria predicha: ",
          np.argmax(model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)[0]) + 1)

    # visualizamos la evolución del error de entrenamiento
    visualizaSerieDatos(history.history['accuracy'], 'Epoch', 'Precisión')
    visualizaSerieDatos(history.history['loss'], 'Epoch', 'Error')

    # visualizamos la estructura del modelo usado
    model.summary()

    # Obtenemos la matriz de confusion para los datos de test
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    confusion = np.zeros((8, 8))
    for i in range(len(y_test)):
        confusion[y_test[i]][y_pred[i]] += 1
    print('Matriz de confusión obtenida:')
    print(confusion)

