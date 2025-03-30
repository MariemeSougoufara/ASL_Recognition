from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import json
import numpy as np
from PIL import Image
from db import init_db, create_user, get_user_by_username
from werkzeug.utils import secure_filename
import tensorflow as tf
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
app = Flask(__name__)
app.secret_key = "super_secret_key"

# Initialisation de la base de données
init_db()

# ====================
# ROUTES FLASK
# ====================

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        if create_user(username, password):
            flash("Inscription réussie.")
            return redirect(url_for('login'))
        else:
            flash("Nom d'utilisateur déjà pris.")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user_by_username(username)
        if user and check_password_hash(user[2], password):
            session['user'] = user[1]
            return redirect(url_for('home'))
        else:
            flash("Identifiants incorrects.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Déconnecté.")
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('login'))


# ================
# PRÉDICTION IA
# ================
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    # Chargement du modèle de prédiction
    # === PARAMÈTRES IDENTIQUES À CEUX DE L'ENTRAÎNEMENT ===
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    num_classes = 29

    # Recréer le modèle avec la même architecture
    def create_model():
        EffNetB0 = EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
        EffNetB0.trainable = True

        model = Sequential([
            EffNetB0,
            BatchNormalization(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # Création et chargement du modèle
    print("Création du modèle...")
    model = create_model()
    model.load_weights("asl_model.h5", by_name=True, skip_mismatch=True)

    UPLOAD_FOLDER = 'static/uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Chargement des classes
    print("Chargement des classes...")
    with open("static/classes.json", "r") as f:
        class_indices = json.load(f)

    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('Aucun fichier trouvé')
            return redirect(request.url)

        image = request.files['image']

        if image.filename == '':
            flash('Aucun fichier sélectionné')
            return redirect(request.url)

        if image:
            filename = secure_filename(image.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(image_path)

            try:
                # Prétraitement identique à celui de l'entraînement
                img = Image.open(image_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prédiction
                result = model.predict(img_array)

                # Classe avec la probabilité la plus élevée
                predicted_class_index = np.argmax(result[0])
                predicted_label = class_indices.get(str(predicted_class_index), f"Classe {predicted_class_index}")
                confidence = float(result[0][predicted_class_index])

                print(f"Prediction: {predicted_label} ({predicted_class_index}) - {confidence:.4f}")

                return render_template('prediction_result.html',
                                       prediction=predicted_label,
                                       confidence=confidence * 100,
                                       image=filename,
                                       index=predicted_class_index)

            except Exception as e:
                print(f"ERREUR: {str(e)}")
                flash(f"Erreur lors du traitement de l'image: {str(e)}")
                return redirect(request.url)

    return render_template('prediction.html')
@app.route('/prediction_result')
def prediction_result():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = request.args.get('prediction', 'Aucune prédiction')
    confidence = request.args.get('confidence', '0')
    try:
        confidence = float(confidence)
    except:
        confidence = 0
    image = request.args.get('image', '')
    index = request.args.get('index', '')

    return render_template('prediction_result.html',
                           prediction=prediction,
                           confidence=confidence,
                           image=image,
                           index=index)

# ================
# ATTAQUE IA
# ================

@app.route('/adversarial', methods=['GET', 'POST'])
def adversarial():
    # Chargement du modèle de prédiction
    # === PARAMÈTRES IDENTIQUES À CEUX DE L'ENTRAÎNEMENT ===
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    num_classes = 29

    # Recréer le modèle avec la même architecture
    def create_model():
        EffNetB0 = EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
        EffNetB0.trainable = True

        model = Sequential([
            EffNetB0,
            BatchNormalization(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # Création et chargement du modèle
    print("Création du modèle...")
    model = create_model()
    model.load_weights("asl_model.h5", by_name=True, skip_mismatch=True)

    UPLOAD_FOLDER = 'static/uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Chargement des classes
    print("Chargement des classes...")
    with open("static/classes.json", "r") as f:
        class_indices = json.load(f)

    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        image = request.files['image']
        epsilon = float(request.form.get('epsilon', 0.05))

        if image:
            filename = secure_filename(image.filename)
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            image_path = os.path.join(upload_folder, filename)
            image.save(image_path)

            # Prétraitement
            img = Image.open(image_path).resize((224, 224)).convert('RGB')
            img_array = np.array(img) / 255.0
            img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)

            # Prédiction initiale
            original_result = model.predict(img_tensor)
            original_class_index = int(np.argmax(original_result))
            original_prediction = class_indices.get(str(original_class_index), f"Classe {original_class_index}")
            original_confidence = round(float(np.max(original_result)) * 100, 2)
            original_image_url = url_for('static', filename=f'uploads/{filename}')

            # --- FGSM ---
            loss_object = tf.keras.losses.CategoricalCrossentropy()
            label_tensor = tf.expand_dims(tf.one_hot(original_class_index, depth=len(class_indices)), axis=0)

            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                prediction = model(img_tensor)
                loss = loss_object(label_tensor, prediction)

            gradient = tape.gradient(loss, img_tensor)
            signed_grad = tf.sign(gradient)
            adversarial_tensor = img_tensor + epsilon * signed_grad
            adversarial_tensor = tf.clip_by_value(adversarial_tensor, 0, 1)

            # Prédiction sur l'image modifiée
            adversarial_result = model.predict(adversarial_tensor)
            adversarial_class_index = int(np.argmax(adversarial_result))
            adversarial_prediction = class_indices.get(str(adversarial_class_index), f"Classe {adversarial_class_index}")
            adversarial_confidence = round(float(np.max(adversarial_result)) * 100, 2)

            # Sauvegarde de l'image perturbée
            adv_img = (adversarial_tensor.numpy()[0] * 255).astype(np.uint8)
            adv_pil = Image.fromarray(adv_img)
            adv_filename = f"adv_{filename}"
            adv_path = os.path.join(upload_folder, adv_filename)
            adv_pil.save(adv_path)
            adversarial_image_url = url_for('static', filename=f'uploads/{adv_filename}')

            # Vérification de l'efficacité de l'attaque
            attack_effective = original_class_index != adversarial_class_index

            return render_template('adversarial_result.html',
                                   original_image=original_image_url,
                                   adversarial_image=adversarial_image_url,
                                   original_prediction=original_prediction,
                                   original_confidence=original_confidence,
                                   adversarial_prediction=adversarial_prediction,
                                   adversarial_confidence=adversarial_confidence,
                                   attack_effective=attack_effective,
                                   epsilon = epsilon)

    return render_template('adversarial.html')


# ================
# DÉFENSE IA
# ================

@app.route('/defense', methods=['GET', 'POST'])
def defense():
    # === PARAMÈTRES IDENTIQUES À L'ENTRAÎNEMENT DU MODÈLE ROBUSTE ===
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    num_classes = 29

    # Architecture du modèle robuste
    def create_defense_model():
        EffNetB0 = EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
        EffNetB0.trainable = True

        model = Sequential([
            EffNetB0,
            BatchNormalization(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    print("Chargement du modèle robuste de défense...")
    model = create_defense_model()
    model.load_weights("robust_asl_model.h5", by_name=True, skip_mismatch=True)

    UPLOAD_FOLDER = 'static/uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    print("Chargement des classes...")
    with open("static/classes.json", "r") as f:
        class_indices = json.load(f)

    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('Aucun fichier trouvé')
            return redirect(request.url)

        image = request.files['image']

        if image.filename == '':
            flash('Aucun fichier sélectionné')
            return redirect(request.url)

        if image:
            filename = secure_filename(image.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(image_path)

            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                result = model.predict(img_array)

                predicted_class_index = np.argmax(result[0])
                predicted_label = class_indices.get(str(predicted_class_index), f"Classe {predicted_class_index}")
                confidence = float(result[0][predicted_class_index])

                print(f"[DEFENSE] Prediction: {predicted_label} ({predicted_class_index}) - {confidence:.4f}")

                return render_template('defense_result.html',
                                       prediction=predicted_label,
                                       confidence=confidence * 100,
                                       image=filename,
                                       index=predicted_class_index)

            except Exception as e:
                print(f"ERREUR (defense): {str(e)}")
                flash(f"Erreur lors du traitement de l'image: {str(e)}")
                return redirect(request.url)

    return render_template('defense.html')

@app.route('/defense_result')
def defense_result():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = request.args.get('prediction', 'Aucune prédiction')
    confidence = request.args.get('confidence', '0')
    try:
        confidence = float(confidence)
    except:
        confidence = 0
    image = request.args.get('image', '')
    index = request.args.get('index', '')

    return render_template('defense_result.html',
                           prediction=prediction,
                           confidence=confidence,
                           image=image,
                           index=index)

