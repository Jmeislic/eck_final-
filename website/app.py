from flask import Flask, render_template, request
from logic import predict_moral_status
# chat gpt promt I am confused I have a seperate file for the front end in html how do i combine it with this also how do I get this to work I want the screen to request an input then i take the input run it through a python function in a seperate file then get an answer and display that on the screen from flask import Flask, render_template app = Flask(__name__) # Define a route for the root URL ('/') @app.route('/') def index(): # Fetch data from the database and prepare for rendering data = get_data_from_database() # Replace this with your actual data retrieval logic # Render the 'index.html' template and pass the retrieved data for rendering return render_template('index.html', data=data) # Placeholder for fetching data from the database def get_data_from_database(): # Replace this function with your actual logic to retrieve data from the database # For now, returning a sample data return {'message': 'Hello, data from the database!'} if __name__ == '__main__': # Run the Flask application app.run(debug=True) <!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0"> <title>Flask App</title> </head> <body> <h1>Data from the Database</h1> <p>{{ data ss }}</p> <!-- Use the 'data' variable in the template --> </body> </html>
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = ""
    norm = ""
    sit = ""
    intent = ""
    action = ""
    if request.method == 'POST':
        user_input = request.form.get("user_input")
        result, norm, sit, intent, action = predict_moral_status(user_input)  # Call your Python function

    return render_template('index.html', result=result, norm=norm, sit=sit, intent=intent, action=action, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
