from flask import Flask, request, jsonify
import boto3
import os
import uuid
import requests
import tempfile 



app = Flask(__name__)

# Configuración de AWS S3
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = 'iaspac'
IA_URL = 'http://<IA_HOST>/analyze'  # Reemplaza <IA_HOST> con el URL de tu servicio de IA en Railway

allowed_extensions = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_unique_filename(filename):
    """Genera un nombre de archivo único usando UUID."""
    extension = filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{extension}"
    return unique_filename

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        file_name = generate_unique_filename(file.filename)
        try:
            s3.upload_fileobj(file, BUCKET_NAME, f'uploads/{file_name}')
            return jsonify({'message': 'File uploaded successfully', 'file_name': file_name}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze/<filename>', methods=['GET'])
def analyze_file(filename):
    try:
        # Descargar la imagen desde S3
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            s3.download_file(BUCKET_NAME, f'uploads/{filename}', temp_path)

        # Enviar la imagen a la IA para su análisis
        with open(temp_path, 'rb') as f:
            response = requests.post(f'{IA_URL}', files={'file': f})

        # Eliminar el archivo temporal
        os.remove(temp_path)

        # Devolver los resultados
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({'error': 'Failed to analyze image'}), response.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)