#Ejemplo de un clasificador de texto usando una red LSTM con embeddings
#Esta red codifica la semántica de las palabras en función a su contexto
#La red LSTM añade procesamiento secuencial de las frases, lo que le permite aprender features
#internas que utiliza para generar la clasificación deseada
import os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import set_random_seed
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, LSTM
from clasificadorLecturaYProcesamientoDatos import lecturaDatosEntrenamientoYTestClasificador, visualizaSerieDatos

#Definición del modelo usado, embeddings, una red lstm, una densa para procesar el resultado del LSTM
#y una final para clasificar en las categorias deseadas
def createModel(tamVoc,tamFrase,tamEmbd):
    model = Sequential()
    model.add(Embedding(tamVoc, tamEmbd, input_length=tamFrase))
    model.add(LSTM(32))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='CategoricalCrossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    return model

# Cargamos los datos y el modelo. Luego lo entrenamos y evaluamos
# El uso de embeddings y el contexto de la lstm producen resultados razonables
# Puedes cambiar el valor de verbose a 1 si quiere ver el proceso de entrenamiento
if __name__ == '__main__':
    set_random_seed(0)
    X_entren, y_entren, X_test, y_test, tamVoc = lecturaDatosEntrenamientoYTestClasificador()

    model=createModel(tamVoc,len(X_entren[0]),  50)
    history = model.fit(X_entren, y_entren, epochs=10, validation_steps=10, batch_size=64 , verbose=0)

    # para hacer pruebas con el modelo entrenado sin tener que reentrenarlo continuamente puedes guardar el
    # modelo y posteriormente cargarlo
    model.save('datos/modelo_entrenado_clasificador_LSTM.h5')
    model = load_model('datos/modelo_entrenado_clasificador_LSTM.h5')

    # Ejemplo de evaluación y clasificación
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