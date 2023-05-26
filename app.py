from pathlib import Path
import json
from flask import Flask,render_template,request,redirect,flash,jsonify
from werkzeug.utils import secure_filename
import os 
from run_pred import *


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
STATIC_FOLDER = 'static'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'webp','tif','tiff','bmp'])
 
def allowed_file(filepath):
    return '.' in filepath and filepath.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# routes
@app.route('/')
def home():
    return render_template('index.html')

check_files = [files for files in os.listdir(app.config['UPLOAD_FOLDER']) if files.lower().endswith(('.jpg','.png', '.jpg', '.jpeg', '.webp','.bmp'))] 
for file in check_files:
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        # return 'file uploaded successfully'

        prediction_op = describe_image(path)
        # try:
        #     os.remove(path)
        # except Exception:
        #     pass
        return render_template('result.html', image=filename, data=prediction_op)
        
    else:
        #flash('Allowed image types are - png, jpg, jpeg')
        #return redirect(request.url)
        error ={}
        error['Message'] = 'Allowed image types are - png, jpg, jpeg, webp, tif, tiff, bmp'
        return json.dumps(error)
    

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)