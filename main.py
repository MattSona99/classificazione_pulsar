from flask import Flask, render_template, request, url_for 
from pyswip import Prolog
from functions import average_stats, validation, classification

app = Flask(__name__)
matrix = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global matrix
    
    if request.method == 'POST':
        data = [
            float(request.form['mean_integrated']),
            float(request.form['std_integrated']),
            float(request.form['kurtosis_integrated']),
            float(request.form['skewness_integrated']),
            float(request.form['mean_dm_snr']),
            float(request.form['std_dm_snr']),
            float(request.form['kurtosis_dm_snr']),
            float(request.form['skewness_dm_snr']),
        ]

        predicted_class = classification(data)

        if predicted_class == 1:
            result_text = "Classified object: Pulsar"
            result_image = url_for('static', filename='pulsar.jpeg')
        else:
            result_text = "Classified object: Neutron Star"
            result_image = url_for('static', filename='neutron.jpg')
            
        return render_template('index.html', 
                                result_text=result_text,
                                result_image=result_image,
                                matrix = matrix)
        
    return render_template('index.html', 
                           result_text="", 
                           result_image="",
                           matrix=matrix)
    

if __name__ == "__main__":
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        # Validazione con chunks
        prolog = Prolog()
        prolog.query("set_prolog_flag(stack_limit, 3*10**9).")
        stats = validation(prolog)
        average_stats(stats, matrix)

    app.run(debug=True)