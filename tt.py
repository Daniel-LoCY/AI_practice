from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('C:\\Users\\cylo\\Downloads', 'test.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)