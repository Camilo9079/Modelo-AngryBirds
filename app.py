from flask import Flask, request, jsonify , send_from_directory
from flask_cors import CORS
import os
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
modelt = load_model(os.path.join(dirname, 'final_model.keras'))
#modelt = custom_vgg_model

app = Flask(__name__)
CORS(app)

# Set the folder where you want to save the uploaded images
UPLOAD_FOLDER = './uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/', methods=['GET'])
def get_example():
    data = {"message": "Este es un ejemplo de respuesta GET"}
    return jsonify(data)

@app.route('/', methods=['POST'])
def upload_image():
    # Check if the POST request contains the 'file' part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # If no file was selected for uploading
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if the file is an allowed image type (optional)
    if file and allowed_file(file.filename):
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Ruta de la imagen de prueba
        # Leer la imagen, cambiar tamaño y preprocesar
        imaget=cv2.resize(cv2.imread(filepath), (224, 224), interpolation = cv2.INTER_AREA)
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
        print(f'Clase predicha: {predicted_class_name}, Porcentaje de confianza: {confidence_percentage:.2f}%')

        # Mostrar la imagen
        #plt.imshow(cv2.cvtColor(np.asarray(imaget), cv2.COLOR_BGR2RGB))
        #plt.axis('off')
        #plt.show()
        return jsonify({"message": f'Clase predicha: {predicted_class_name}, Porcentaje de confianza: {confidence_percentage:.2f}%', "file_path": filepath}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Recurso no encontrado"}), 404

@app.route('/')
def serve_interface():
    return send_from_directory('.', 'index.html')

# Ejecutamos la app
if __name__ == '__main__':
    app.run(debug=True)