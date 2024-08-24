import cv2
import mediapipe as mp
import numpy as np
import os
from flask import Flask, request, jsonify
import boto3
import tempfile

app = Flask(__name__)

# Inicializar Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Configuración de AWS S3
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = 'iaspac'

def detect_dark_areas(region):
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    alpha = 1.5
    beta = -50
    adjusted_region = cv2.convertScaleAbs(gray_region, alpha=alpha, beta=beta)
    blurred_region = cv2.GaussianBlur(adjusted_region, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_region, 60, 255, cv2.THRESH_BINARY_INV)
    dark_areas = cv2.countNonZero(thresh)
    total_area = region.shape[0] * region.shape[1]
    percentage_oje = (dark_areas / total_area) * 100
    return percentage_oje

def detect_wrinkles(region):
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    alpha = 2.0
    beta = -50
    adjusted_region = cv2.convertScaleAbs(gray_region, alpha=alpha, beta=beta)
    blurred_region = cv2.GaussianBlur(adjusted_region, (1, 1), 0)
    edges = cv2.Canny(blurred_region, 50, 150)
    _, thresh = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)
    wrinkles = cv2.countNonZero(thresh)
    total_area = region.shape[0] * region.shape[1]
    percentage_arr = (wrinkles / total_area) * 100
    return percentage_arr

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Obtener la imagen de la solicitud
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Guardar la imagen temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        # Leer la imagen
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'error': 'No se pudo cargar la imagen'}), 500

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                eye_indices_profile = [53, 160, 445, 355]
                face_indices_profile = [10, 234, 454, 150]
                eye_points_profile = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in eye_indices_profile]
                face_points_profile = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in face_indices_profile]
                eye_roi_profile = image[min(p[1] for p in eye_points_profile):max(p[1] for p in eye_points_profile),
                                        min(p[0] for p in eye_points_profile):max(p[0] for p in eye_points_profile)]
                face_roi_profile = image[min(p[1] for p in face_points_profile):max(p[1] for p in face_points_profile),
                                         min(p[0] for p in face_points_profile):max(p[0] for p in face_points_profile)]
                ojeras = detect_dark_areas(eye_roi_profile)
                arrugas = detect_wrinkles(face_roi_profile)
                resultado = (ojeras + arrugas) / 2
                results = {
                    'ojeras': ojeras,
                    'arrugas': arrugas,
                    'resultado': resultado
                }
                return jsonify(results), 200
        return jsonify({'error': 'No face landmarks detected'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
