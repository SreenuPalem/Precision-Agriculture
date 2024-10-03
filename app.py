import bcrypt
from flask import Flask, redirect, render_template, url_for, request, flash
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt  # Add this import statement
import pandas as pd
import pickle
import numpy as np
import config
import requests
import torch
from utils.model import ResNet9
import firebase_admin
from firebase_admin import credentials, db
from torchvision import transforms
from PIL import Image
import io

# Your code here
# Loading ML model
with open('models/RForest.pkl', 'rb') as file:
    model = pickle.load(file)

cred = credentials.Certificate("C:/Users/P SREENIVAS REDDY/Downloads/btproject-949d4-firebase-adminsdk-4e3jm-54d2768107.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://btproject-949d4-default-rtdb.firebaseio.com/'
})

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# disease prediction
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust this size to what your model expects
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Optional normalization
])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config["SECRET_KEY"] = 'thisissecretkey'

class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password

# Sample users dictionary for demonstration
users = {
    'john': bcrypt.generate_password_hash('password1').decode('utf-8'),
    'emma': bcrypt.generate_password_hash('password2').decode('utf-8')
}

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "username"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "password"})
    submit = SubmitField("Register")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "username"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "password"})
    submit = SubmitField("Login")

class AdminRegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "username"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "password"})
    submit = SubmitField("Register")

@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():
    form = LoginForm()  # Create an instance of the LoginForm class
    if form.validate_on_submit():
        # Form is submitted and validated
        username = form.username.data
        password = form.password.data

        # Placeholder logic for checking username and password
        if username == 'admin' and password == 'password':
            # Successful login
            flash('Login successful!', 'success')
            # Redirect to admin dashboard
            return redirect(url_for('Admindashboard'))
        else:
            # Invalid username or password
            flash('Invalid username or password. Please try again.', 'danger')

    # If form is not submitted or validation fails, render the login page with the form
    return render_template("admindashboard.html", form=form)


@app.route("/adminsignup", methods=['GET', 'POST'])
def adminsignup():
    form = AdminRegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Add the new admin to the users dictionary
        users[username] = hashed_password

        flash('Admin account created successfully.', 'success')
        return redirect(url_for('admindashboard'))  # Update this line

    return render_template("adminsignup.html", form=form)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    # Your contact form handling code here
    return render_template("contact.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users and bcrypt.check_password_hash(users[username], password):
            flash('Login successful!', 'success')
            # Redirect to the dashboard or other page after successful login
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template("login.html", form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # Your dashboard code here
    return render_template("dashboard.html")

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    # Your logout code here
    return redirect(url_for('hello_world'))

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    form = RegisterForm()  # Create an instance of the RegisterForm class
    if form.validate_on_submit():
        # Extract username and password from the form
        username = form.username.data
        password = form.password.data
        
        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Add the new user to the users dictionary
        users[username] = hashed_password
        
        flash('Account created successfully. Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template("signup.html", form=form)

@app.route('/crop-recommend')
def crop_recommend():
    # Your crop recommendation code here
    return render_template("crop.html")

@app.route('/fertilizer')
def fertilizer_recommendation():
    # Your fertilizer recommendation code here
    return render_template("fertilizer.html")

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = '- Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)
        
        file = request.files.get('file')
        if not file:
            flash('No file selected.', 'danger')
            return render_template('disease.html', title=title)
        
        try:
            img = file.read()

            # Predict the disease using the model
            prediction = predict_image(img)

            # Render the prediction result on a new page
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            # Print the exception to the console for debugging
            print(f"Error during prediction: {e}")
            flash('An error occurred during prediction. Please try again.', 'danger')
            return redirect(request.url)
    
    return render_template('disease.html', title=title)

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    # Your crop prediction code here
    title = '- Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph_ref = db.reference('/ph')  # Use reference() method to specify the path
        ph = ph_ref.get()

        rainfall_ref = db.reference('/rainfall')  # Use reference() method to specify the path
        rainfall = rainfall_ref.get()

        temperature_ref = db.reference('/temperature')  # Use reference() method to specify the path
        temperature = temperature_ref.get()

        humidity_ref = db.reference('/humidity')  # Use reference() method to specify the path
        humidity = humidity_ref.get()
      
        
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction, title=title)

    else:
        return render_template('try_again.html', title=title)

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = '- Fertilizer Suggestion'

    fertilizer_dic = {
    'NHigh': {
        'suggestions': 'Reduce nitrogen-rich fertilizers, focus on balanced fertilizers.',
        'fertilizer_names': ['NitroMax', 'NitroGrow Plus']
    },
    'NLow': {
        'suggestions': 'Use fertilizers with higher nitrogen content.',

        'fertilizer_names': ['N-Power Boost', 'GreenLeaf Grow']
    },
    'PHigh': {
        'suggestions': 'Decrease phosphorus application, opt for phosphorus-balanced fertilizers.',
        'fertilizer_names': ['PhosRich Blend', 'PhosBoost Supreme']
    },
    'PLow': {
        'suggestions': 'Choose fertilizers with higher phosphorus content.',
        'fertilizer_names': ['PhosMax Advance', 'BloomBuilder Pro']
    },
    'KHigh': {
        'suggestions': 'Reduce potassium-rich fertilizers, seek balanced potassium sources.',
        'fertilizer_names': ['K-Power Balanced', 'Potash Plus Mix']
    },
    'KLow': {
        'suggestions': 'Use fertilizers with higher potassium content.',
        'fertilizer_names': ['K-Gro Maximizer', 'PotassiumBoost Elite']
    },
    'phHigh': {
        'suggestions': 'Use acidic fertilizers, consider soil acidifiers.',
        'fertilizer_names': ['AcidicBlend Pro', 'pHLower Solution']
    },
    'phLow': {
        'suggestions': 'Choose alkaline or neutral fertilizers, incorporate lime.',
        'fertilizer_names': ['AlkaMax Plus', 'pHBalance Granules']
    }
}



    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    ph_ref = db.reference('/ph')  # Use reference() method to specify the path
    ph = ph_ref.get()

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]
    phr = df[df['Crop'] == crop_name]['pH'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    h = phr - ph 
    temp = {abs(n): "N", abs(p): "P", abs(k): "K",abs(h): "ph" }
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "NLow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "PLow"
    elif max_value == "K":
        if k < 0:
            key = 'KHigh'
        else:
            key = "KLow"
    else:
        if h < 0:
            key = "phHigh"
        else:
            key = "phLow"

    recommendation = str(fertilizer_dic[key])

    return render_template('fertilizer-result.html', recommendation=recommendation, title=title)

@app.route("/display")
def querydisplay():
    # Your query display code here
    return render_template("display.html")

@app.route("/admindashboard")
def admindashboard():
    # Your admin dashboard code here
    return render_template("admindashboard.html")

@app.route("/reg", methods=['GET', 'POST'])
def reg():
    # Your registration code here
    return render_template("reg.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
