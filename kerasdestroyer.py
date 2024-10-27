
from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


names = ['Anambé Barrado',
'Anambé Cinéreo',
'Anambé Unicolor',
'Cabezón Alas Blancas',
'Cabezón Canelo',
'Cabezón sp.',
'Cimerillo Andino',
'Cotinga Cresticastaño',
'Cotinga Crestirrojo',
'Curutié Colorado',
'Frutero Barrado',
'Frutero Pechidorado',
'Frutero Verdinegro',
'Gallito de las Rocas Peruano',
'Guardabosques Oscuro',
'Moscareta Colinegra',
'Mosquerito Espatulilla Común',
'Mosquerito Espatulilla Gris',
'Mosquerito Ocre',
'Mosquerito Ojiblanco',
'Mosquerito Ojos Blancos',
'Mosquerito Piquicurvo Sureño',
'Mosquero Rayadito',
'Mosqueta Capirotada',
'Orejerito Antioqueño',
'Orejerito Jaspeado',
'Orejerito Variegado',
'Orejero Coronigrís',
'Orejero Pechirrufo',
'Picoplano Bigotudo',
'Picoplano Equinoccial',
'Picoplano Pechirrufo',
'Pijuí Pechiblanco',
'Piprites Verde',
'Piscuiz Barbiblanco',
'Saltarín Alidorado',
'Saltarín Amarillo',
'Saltarín Barbiblanco',
'Saltarín Cabecidorado',
'Saltarín Coroniazul (velutina/minuscula)',
'Saltarín Coroniblanco',
'Saltarín Gorjiblanco Oriental',
'Saltarín Lanceolado',
'Saltarín Rayado Occidental',
'Tiranuelo Bronceado',
'Tiranuelo Cabecirrojo',
'Titira Pico Negro',
'Titira Puerquito',
'Titirijí Cabecinegro',
'Titirijí Capirrufo',
'Titirijí Gorjinegro',
'Titirijí Perlado',
'Yacutoro']

# Cargar el modelo
dirname = os.path.dirname(__file__)
modelt = load_model(os.path.join(dirname, 'Modelosmodel_VGG16_v6.keras'))
#modelt = custom_vgg_model

# Ruta de la imagen de prueba
imaget_path = os.path.join(dirname, 'uploaded_images\gustavo bladimir.png')

# Leer la imagen, cambiar tamaño y preprocesar
imaget=cv2.resize(cv2.imread(imaget_path), (224, 224), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)

# Obtener las predicciones del modelo
preds = modelt.predict(xt)

# Obtener la clase predicha y su porcentaje de confianza
predicted_class_index = np.argmax(preds)
predicted_class_name = names[predicted_class_index]
confidence_percentage = preds[0][predicted_class_index] * 100

# Imprimir el resultado
print(f'Clase predicha: {predicted_class_name}')
print(f'Porcentaje de confianza: {confidence_percentage:.2f}%')

# Mostrar la imagen
plt.imshow(cv2.cvtColor(np.asarray(imaget), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()