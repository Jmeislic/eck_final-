from flask import Flask, render_template, request
from logic import predict_moral_status

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        user_input = request.form.get("user_input")
        result = predict_moral_status(user_input)  # Call your Python function

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
