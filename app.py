from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = joblib.load('genre_prediction_model.joblib')

# HTML template para la página web
html_template = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Plot Genre Predictor</title>
  </head>
  <body>
    <div style="text-align: center; margin-top: 50px;">
      <h1>Plot Genre Predictor</h1>
      <form action="/predict" method="get">
        <textarea name="plot" rows="10" cols="50" placeholder="Enter the plot here"></textarea><br><br>
        <input type="submit" value="Predict Genre">
      </form>
      {% if genre %}
        <h2>Predicted Genre: {{ genre }}</h2>
      {% endif %}
    </div>
  </body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['GET'])
def predict():
    # Obtener el parámetro "plot" de la solicitud GET
    plot = request.args.get('plot')
    
    if not plot:
        return jsonify({'error': 'Plot parameter is required'}), 400
    
    # Realizar la predicción
    prediction = model.predict([plot])[0]
    
    # Renderizar el template con la predicción
    return render_template_string(html_template, genre=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
