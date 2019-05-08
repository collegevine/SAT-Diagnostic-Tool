from flask import Flask, render_template, request, send_from_directory
from analysis import calculate_math_score, calculate_verbal_score, run_analysis, plot_verbal, plot_math, plot_improve_barchart, plot_math_pie, plot_verbal_pie, plot_total_miss_barchart


import os

app = Flask(__name__, template_folder = "./templates")

@app.route('/')
def index():
    return render_template('home.html', my_list=range(1,41))
@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('img', path)
@app.route('/img-math/<path:path>')
def send_math(path):
    return send_from_directory('img-math', path)

@app.route('/SATAnalysis', methods=['POST'])
def success():
    try:
        if request.method == 'POST':
            obj = run_analysis(request.form)
            tables = {
                    'm_diff'    : obj.get('math_difficulty'),
                    'm_concept' : obj.get('math_concepts'),
                    'r_diff'    : obj.get('reading_difficulty'),
                    'r_concept' : obj.get('reading_concepts'),
                    'w_diff'    : obj.get('writing_difficulty'),
                    'w_concept' : obj.get('writing_concepts'),
                    'r_explain' : obj.get('reading_explain'),
                    'w_explain' : obj.get('writing_explain'),
                    'm_explain' : obj.get('math_explain'),
                    'm_improve' : obj.get('math_improve'),
                    'v_improve' : obj.get('verbal_improve'),
                    'v_best_concepts' : obj.get('verbal_best_concepts'),
                    'm_best_concepts' : obj.get('math_best_concepts'),
                    'v_worst_concepts' : obj.get('verbal_worst_concepts'),
                    'm_worst_concepts' : obj.get('math_worst_concepts'),
                    'm_improve_stmt' : obj.get('math_improve_stmt'),
                    'v_improve_stmt' : obj.get('verbal_improve_stmt'),
            }

            verbal_plot = plot_verbal(obj)
            math_plot = plot_math(obj)
            math_improve_plot = plot_total_miss_barchart(obj, "math_concept_plot", "Math")
            verbal_improve_plot = plot_total_miss_barchart(obj, "verbal_concept_plot", "Verbal")
            #print(obj.get('math_concepts'))
            return render_template('result.html',
                    table = tables,
                    obj = obj,
                    cols_concept = ['', 'Concept', 'Wrong', 'Percent of Total Test Questions'],
                    cols_diff = ['','Question Level Difficulty', 'Wrong', 'Percent of Total Questions'],
                    cols_explain = ['', 'Section','Question','Your Answer', 'Correct Answer', 'Explaination'],
                    cols_improve = ['', 'Concept', 'Possible Score Increase'],
                    verbal_improve_plot = verbal_improve_plot,
                    math_improve_plot = math_improve_plot,
                    verbal_plot = verbal_plot,
                    math_plot = math_plot)

        else:
            pass
    except Exception as e:
        return render_template('500.html', error = str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port, debug = True)
