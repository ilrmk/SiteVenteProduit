from flask import Flask, render_template, request
import secrets
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import cv2



app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    # Votre code Python pour générer des données
    products_html = ""
    category = os.listdir('static/data')

    for cat in category:
        images = os.listdir(f'static/data/{cat}')
        products_html += f"""
                        <div class="categorie-titre">{cat}</div><br>
                        <div class="categorie-contenu">
                        """
        for img in images:
            m = cv2.imread(f'static/data/{cat}/{img}')
            if m is not None:
                # L'image est valide, vous pouvez maintenant la redimensionner
                new_size = (225, 225)
                m = cv2.resize(m, new_size, interpolation=cv2.INTER_AREA)
                ext=img.split(".")
                cv2.imwrite(f'static/data/{cat}/{ext[0]+"."+ext[1]}.jpg', m)
                
            products_html += f"""
                            <section class="produit">
                                <img src="/static/data/{cat}/{img}" alt="{cat} - {img}" />
                                <h1>{ext[0]}</h1>
                                <p>{ext[1]}</p>
                            </section>
                            """
        products_html += "</div>"

    # Récupérer l'argument products_html de la requête
    return render_template('index.html', products_html=products_html)


@app.route('/chercher', methods=['POST'])
def submit():
    products_html=""
    if request.method == 'POST':
        uploaded_file = request.files['chooseImageButton']

        # Vérifier si un fichier a été téléchargé
        if uploaded_file.filename != '':
            random_name = secrets.token_hex(5) 
            uploaded_file.save(f'uploads/{random_name}.jpg')

            model = ResNet50(weights='imagenet')

            img_path = f'uploads/{random_name}.jpg'

            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            result = decode_predictions(preds, top=3)[0]
            result = result[0][1]
            etat=0
            
            if result is not None:
                category = os.listdir('static/data')

                products_html = "<div class='affiche_des_produits'>"
                for cat in category:
                    if cat == result:
                        etat=1
                        images = os.listdir(f'static/data/{cat}')
                        for img in images:

                            ext=img.split(".")
                            products_html += f"""
                                    <section class="produit">
                                        <img src="/static/data/{cat}/{img}" alt="{cat} - {img}" />
                                        <h1>{ext[0]}</h1>
                                        <p>{ext[1]}</p>
                                        
                                    </section>
                                    
                                    """
                        products_html +="</div>"
                        break
            if etat==0:
                products_html =f"""
                    <div class="container text-center">
	                <div class="sad-animation">
		            <h1 class="display-4 custom-text-color">Pas de produit comme vous cherchez 😢</h1>
	                </div>
                    </div>
                """

        else:
           
            products_html = f"""
                    <div class="container text-center">
	                <div class="sad-animation">
		            <h1 class="display-4 custom-text-color">Essayer de reimporter une image 😢</h1>
	                </div>
                    </div>
                """
            
        # Rediriger vers la page index avec l'argument products_html
    return render_template('index.html', products_html=products_html)


if __name__ == '__main__':
    app.run(debug=True)
