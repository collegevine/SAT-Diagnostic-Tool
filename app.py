from flask import Flask, render_template, request, send_from_directory
from analysis import calculate_math_score, calculate_verbal_score, run_analysis, format_profile_table, plot_verbal, plot_math

import os

app = Flask(__name__, template_folder = "./templates")

@app.route('/home')
def index():
    return render_template('home.html', my_list=range(1,41))
@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('img', path)

@app.route('/SATAnalysis', methods=['POST'])
def success():
    try:
        if request.method == 'POST':
            obj = run_analysis(request.form)

            tables = {
                    'm_diff' : obj.get('math_difficulty'),
                    'm_concept' : obj.get('math_concepts'),
                    'r_diff' : obj.get('reading_difficulty'),
                    'r_concept': obj.get('reading_concepts'),
                    'w_diff' : obj.get('writing_difficulty'),
                    'w_concept' : obj.get('writing_concepts')
            }

            verbal_plot = plot_verbal(obj)
            math_plot = plot_math(obj)

            return render_template('result.html',
                    table = tables,
                    obj = obj,
                    cols_concept = ['', 'Concept', 'Percent', 'Wrong', 'Total'],
                    cols_diff = ['', 'Difficulty', 'percent', 'Wrong', 'Total'],
                    verbal_plot = verbal_plot,
                    math_plot = math_plot)

        else:
            pass
    except Exception as e:
        return render_template('500.html', error = str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port, debug = True)
