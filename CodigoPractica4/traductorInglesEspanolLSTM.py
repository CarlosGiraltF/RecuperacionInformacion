#Ejemplo de un traductor de texto usando una red LSTM con embeddings
#Esta red codifica la semántica de las palabras en función a su contexto
#La red LSTM añade procesamiento secuencial de las frases, lo que le permite aprender features
#internas que utiliza para generar la clasificación deseada
#La red es demasiado pequeña y se entrena demasiado poco como para resolver apropiadamente el problema,
#solo es un ejemplo simple de como funciona la arquitectura de estos sistemas
import os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import optimizers
from keras.utils import set_random_seed
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from traductorLecturaYProcesamientoDatos import lecturaDatosEntrenamientoYTestTraductor, visualizaSerieDatos

#Definición del modelo usado, embeddings, una red lstm como encoder y otra como decoder.
#La conexión pasa los estados ocultos en cada etapa del decoder. La capa final de clasificación da un valor (palabra)
#para cada etapa del decoder
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(TimeDistributed(Dense(out_vocab, activation='softmax')))
    return model

# Cargamos los datos y el modelo. Luego lo entrenamos y evaluamos
# La red es demasiado pequeña y con demasiadas pocas iteraciones para producir resultados minimamente razonables
# Puedes observar en la ejecición como poco a poco va convergiendo la red. Aunque con las pocas iteraciones
# indicadas, se queda muy lejos de valores desables de precisión
if __name__ == '__main__':
    set_random_seed(0)
    X_entren, y_entren, X_test, y_test, X_tokenizer, y_tokenizer, X_maxlen, y_maxlen = lecturaDatosEntrenamientoYTestTraductor()
    X_tamVoc = len(X_tokenizer.word_index); y_tamVoc = len(y_tokenizer.word_index)
    model = define_model(X_tamVoc, y_tamVoc, X_maxlen, y_maxlen, 64)
    rms = optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    history = model.fit(X_entren, y_entren.reshape(y_entren.shape[0], y_entren.shape[1], 1),
                        epochs=10, batch_size=512, validation_split=0.2, verbose=1)

    # para hacer pruebas con el modelo entrenado sin tener que reentrenarlo continuamente puedes guardar el
    # modelo y posteriormente cargarlo
    model.save('datos/modelo_entrenado_traductor_LSTM.h5')
    model = load_model('datos/modelo_entrenado_traductor_LSTM.h5')

    # Ejemplo de evaluación y clasificación
    scores = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=0)
    print("Precisión del modelo con los test: %.2f%%" % (scores[1] * 100))

    print("Ejemplo de traducción de la componente 3000 del test")
    prediccion = np.argmax(model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)[0], axis=1)
    print("Ingles: ",X_tokenizer.sequences_to_texts([X_test[3000]]))
    print("Español: ", y_tokenizer.sequences_to_texts([y_test[3000]]))
    print("Predicción: ",y_tokenizer.sequences_to_texts([prediccion]))

    # visualizamos la evolución del error de entrenamiento
    visualizaSerieDatos(history.history['accuracy'], 'Epoch', 'Precisión')
    visualizaSerieDatos(history.history['loss'], 'Epoch', 'Error')

    # visualizamos la estructura del modelo usado
    model.summary()