from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

# Cargar los modelos y el vectorizador
modelo = joblib.load('final_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
genres_df = pd.read_csv('genres.csv')

# Convertir la columna de géneros de strings a listas
genres_df['genres'] = genres_df['genres'].astype(str).apply(eval)

# Ajustar el MultiLabelBinarizer con los datos de géneros
le = MultiLabelBinarizer()
le.fit(genres_df['genres'])

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        plot = request.form['plot']
        if not plot:
            return "Please provide a plot.", 400

        # Transformar el plot usando el vectorizador
        plot_vectorizado = vectorizer.transform([plot])

        # Realizar la predicción con el modelo
        probabilidades = modelo.predict_proba(plot_vectorizado)

        # Obtener los nombres de los géneros
        generos = le.classes_

        # Combinar los géneros con sus probabilidades
        resultados = list(zip(generos, probabilidades[0]))

        # Ordenar los resultados por probabilidad descendente
        resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)

        # Convertir a DataFrame para facilitar la visualización
        resultados_df = pd.DataFrame(resultados_ordenados, columns=["Género", "Probabilidad"])

        # Convertir a HTML
        resultados_html = resultados_df.to_html(index=False)

        # Plantilla HTML para la visualización
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Géneros y Probabilidades</title>
        </head>
        <body>
            <h1>Géneros y Probabilidades</h1>
            <form method="post">
                <label for="plot">Plot:</label><br>
                <textarea id="plot" name="plot" rows="4" cols="50"></textarea><br><br>
                <input type="submit" value="Submit">
            </form>
            <h2>Resultados:</h2>
            {{ table|safe }}
        </body>
        </html>
        """
        return render_template_string(html_template, table=resultados_html)
    
    # Mostrar el formulario en una petición GET
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Géneros y Probabilidades</title>
    </head>
    <body>
        <h1>Géneros y Probabilidades</h1>
        <form method="post">
            <label for="plot">Plot:</label><br>
            <textarea id="plot" name="plot" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    """
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
