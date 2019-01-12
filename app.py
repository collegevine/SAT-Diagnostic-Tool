from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder = "./templates")

@app.route('/home')
def index():
    return render_template('home.html', my_list=range(1,41))

@app.route('/SATAnalysis', methods=['POST'])
def success():
    try:
        if request.method == 'POST':
            print(request.form)
            return render_template('home.html', my_list=range(1,41))

        else:
            pass
    except Exception as e:
        return render_template('500.html', error = str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port, debug = True)
