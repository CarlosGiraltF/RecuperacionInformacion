#Ejemplo de un clasificador de texto usando una red densa
#Es una demostración de por qué las redes simples no funcionan en tareas de tratamiento de texto
import os,  numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import set_random_seed
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense
from clasificadorLecturaYProcesamientoDatos import lecturaDatosEntrenamientoYTestClasificador, visualizaSerieDatos

#La codificación númerica de las palabras generada por el tokenizer la transformamos al rango
#0-1 para poder pasarselo a la red
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#Definición del modelo usado, 2 capas densas y una final de clasificación en las categorias deseadas
def createModel():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='CategoricalCrossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    return model


# Cargamos los datos y el modelo. Luego lo entrenamos y evaluamos
# Este modelo tieme muy mal rendimiento por las caracteristicas de la red.
# La codificación de texto se ha normalizado entre 0 y 1 y la red densa no aporta contexto a cada palabra
# Puedes cambiar el valor de verbose a 1 si quiere ver el proceso de entrenamiento
if __name__ == '__main__':
    set_random_seed(0)
    X_entren, y_entren, X_test, y_test, tamVoc = lecturaDatosEntrenamientoYTestClasificador()
    model=createModel()
    history = model.fit(NormalizeData(X_entren), y_entren, epochs=20, validation_steps=10, batch_size=64 , verbose=0)

    #para hacer pruebas con el modelo entrenado sin tener que reentrenarlo continuamente puedes guardar el
    #modelo y posteriormente cargarlo
    model.save('datos/modelo_entrenado_clasificador_densa.h5')
    model = load_model('datos/modelo_entrenado_clasificador_densa.h5')

    #Ejemplo de evaluación y clasificación
    scores = model.evaluate(NormalizeData(X_test), y_test, verbose=0)
    print("Precisión del modelo con los test: %.2f%%" % (scores[1] * 100))
    print("Ejemplo de clasificación de la componente 0 del test")
    print("Categoria real: ", np.argmax(y_test[0]) +1, " Categoria predicha: ",
          np.argmax(model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)[0]) +1)

    #visualizamos la evolución del error de entrenamiento
    visualizaSerieDatos(history.history['accuracy'], 'Epoch', 'Precisión')
    visualizaSerieDatos(history.history['loss'], 'Epoch', 'Error')

    #visualizamos la estructura del modelo usado
    model.summary()
