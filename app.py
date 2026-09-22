from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import base64
import io
import os

import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_libras.keras')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'classes.npy')
IMG_SIZE = 128
SAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teste_OK')

model = None
classes = None


def load_ai():
    global model, classes
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError('Arquivo modelo_libras.keras não encontrado.')
        if not os.path.exists(CLASSES_PATH):
            raise FileNotFoundError('Arquivo classes.npy não encontrado.')

        model = tf.keras.models.load_model(MODEL_PATH)
        classes = np.load(CLASSES_PATH, allow_pickle=True)


def prepare_image(data_url: str):
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]

    image_bytes = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/documentacao')
def documentacao():
    return render_template('documentacao.html')


@app.route('/quiz')
def index():
    return render_template('index.html')


@app.get('/sample-images')
def sample_images():
    base_dir = SAMPLE_IMAGES_DIR
    if not os.path.isdir(base_dir):
        return jsonify({})

    result = {}
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        files = []
        for filename in sorted(os.listdir(folder_path)):
            if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                version = os.stat(os.path.join(folder_path, filename)).st_mtime_ns
                files.append(url_for('sample_image', letter=folder_name, filename=filename, v=version))

        if files:
            result[folder_name] = files

    return jsonify(result)


@app.get('/audio/<filename>')
def quiz_audio(filename):
    return send_from_directory(os.path.join(app.root_path, 'audio'), filename)


@app.get('/sample-image/<path:letter>/<filename>')
def sample_image(letter, filename):
    return send_from_directory(SAMPLE_IMAGES_DIR, f'{letter}/{filename}')


@app.post('/predict')
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get('image')

        if not image_data:
            return jsonify({'error': 'Nenhuma imagem foi enviada.'}), 400

        load_ai()
        x = prepare_image(image_data)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))

        return jsonify({
            'letter': str(classes[idx]),
            'confidence': float(probs[idx]),
        })

    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 503
    except Exception as exc:
        return jsonify({'error': f'Falha ao processar a imagem: {exc}'}), 500


@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_found': os.path.exists(MODEL_PATH),
        'classes_found': os.path.exists(CLASSES_PATH),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
