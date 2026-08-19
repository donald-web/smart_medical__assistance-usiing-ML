import os
from io import BytesIO
from datetime import datetime
from rapidfuzz import process

import joblib
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from disease_info import DISEASE_INFO

if not os.path.exists("models/disease_model.pkl") or not os.path.exists("models/features.pkl"):
    print("Model files not found. Training model first...")
    from train_model import train_model
    train_model()

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/smart_medical_assistant_db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

model = joblib.load("models/disease_model.pkl")
FEATURES = joblib.load("models/features.pkl")

def extract_symptoms_ai(user_message):
    user_message = user_message.lower()
    extracted = []

    symptom_labels = {
        symptom: symptom.replace("_", " ")
        for symptom in FEATURES
    }

    for symptom, readable in symptom_labels.items():
        if readable in user_message:
            extracted.append(symptom)

    words = user_message.split()

    for word in words:
        match = process.extractOne(
            word,
            list(symptom_labels.values()),
            score_cutoff=75
        )

        if match:
            matched_readable = match[0]

            for symptom, readable in symptom_labels.items():
                if readable == matched_readable and symptom not in extracted:
                    extracted.append(symptom)

    return extracted

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.String(10))
    gender = db.Column(db.String(20))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(120))
    blood_group = db.Column(db.String(10))
    address = db.Column(db.Text)
    parent_name = db.Column(db.String(120))
    parent_phone = db.Column(db.String(20))
    parent_email = db.Column(db.String(120))
    symptoms = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20))
    doctor = db.Column(db.String(120))
    appointment_date = db.Column(db.String(50))
    appointment_time = db.Column(db.String(50))
    reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def get_disease_image(prediction):
    image_name = prediction.lower().replace(" ", "_").replace("/", "_") + ".png"
    path = os.path.join("static", "disease_images", image_name)
    if os.path.exists(path):
        return "disease_images/" + image_name
    return "disease_images/default.png"

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():

    patient_name = request.form.get("patient_name", "").strip()
    age = int(request.form.get("age", 0))

    gender = request.form.get("gender")
    phone = request.form.get("phone")
    email = request.form.get("email")
    blood_group = request.form.get("blood_group")
    address = request.form.get("address")

    parent_name = request.form.get("parent_name")
    parent_phone = request.form.get("parent_phone")
    parent_email = request.form.get("parent_email")

    if age < 1 or age > 120:
        flash("Please enter a valid age between 1 and 120.")
        return redirect(url_for("index"))

    if age < 18:
        if not parent_name or not parent_phone or not parent_email:
            flash("Parent or guardian details are required for patients below 18 years.")
            return redirect(url_for("index"))

    if not patient_name:
        flash("Please enter patient name.")
        return redirect(url_for("index"))

    selected = []
    values = []
    for symptom in FEATURES:
        checked = 1 if request.form.get(symptom) == "on" else 0
        values.append(checked)
        if checked:
            selected.append(symptom.replace("_", " ").title())

    if len(selected) < 5:
       flash("Please select at least 5 symptoms for accurate prediction.")
       return redirect(url_for("index"))

    probabilities = model.predict_proba([values])[0]

    top_predictions = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    prediction = top_predictions[0][0]
    confidence = round(top_predictions[0][1] * 100, 2)

    # Relative confidence = how much more likely than random chance
    random_chance = 100 / len(model.classes_)
    relative_conf = round(confidence / random_chance, 1)

    top_prediction_results = []

    for disease, prob in top_predictions:
        percent = round(prob * 100, 2)

        top_prediction_results.append({
            "disease": disease,
            "percent": percent
        })

    warning_message = None

    if confidence < 10:
           warning_message = "Very low confidence. Please select more symptoms."
    elif confidence < 30:
           warning_message = "Moderate confidence. Consider selecting more symptoms for accuracy."
           
    info = DISEASE_INFO.get(prediction, {
            "description": "Information for this disease is not available yet.",
            "causes": ["Information not available"],
            "symptoms": ["Information not available"],
            "precautions": ["Consult a qualified doctor", "Avoid self-medication", "Monitor symptoms"],
            "treatment": ["Medical consultation recommended"],
            "doctor": "General Physician"
        })

    record = PredictionHistory(
        patient_name=patient_name, age=age, gender=gender, phone=phone,
        email=email, blood_group=blood_group, address=address,
        symptoms=", ".join(selected), prediction=prediction, confidence=f"{confidence}%",
        parent_name=parent_name,
        parent_phone=parent_phone,
        parent_email=parent_email
    )
    db.session.add(record)
    db.session.commit()

    return render_template(
            "result.html", record_id=record.id,
            relative_conf=relative_conf,
            top_prediction_results=top_prediction_results,
            patient_name=patient_name, age=age, gender=gender, phone=phone, email=email,
            blood_group=blood_group, address=address, symptoms=selected,
            prediction=prediction, confidence=confidence, warning_message=warning_message, image_path=get_disease_image(prediction),
            description=info["description"], causes=info["causes"], disease_symptoms=info["symptoms"],
            precautions=info["precautions"], treatment=info["treatment"], doctor=info["doctor"],
            parent_name=parent_name,
            parent_phone=parent_phone,
            parent_email=parent_email,
        )

@app.route("/history")
def history():
    records = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()
    return render_template("history.html", records=records)

@app.route("/metrics")
def metrics():
    text = "Run python train_model.py first."
    if os.path.exists("models/metrics.txt"):
        with open("models/metrics.txt", "r", encoding="utf-8") as f:
            text = f.read()
    return render_template("metrics.html", metrics=text)

@app.route("/dashboard")
def dashboard():
    records = PredictionHistory.query.all()
    disease_count, gender_count = {}, {}
    for r in records:
        disease_count[r.prediction] = disease_count.get(r.prediction, 0) + 1
        if r.gender:
            gender_count[r.gender] = gender_count.get(r.gender, 0) + 1
    return render_template("dashboard.html", disease_count=disease_count, gender_count=gender_count)

@app.route("/download_report/<int:record_id>")
def download_report(record_id):
    r = PredictionHistory.query.get_or_404(record_id)
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle("Medical Prediction Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(160, 800, "Smart Medical Assistant Report")
    pdf.setFont("Helvetica", 11)
    y = 750
    lines = [
        f"Patient Name: {r.patient_name}", f"Age: {r.age}", f"Gender: {r.gender}",
        f"Phone: {r.phone}", f"Email: {r.email}", f"Blood Group: {r.blood_group}",
        f"Address: {r.address}", "", f"Symptoms: {r.symptoms}",
        f"Predicted Disease: {r.prediction}", f"Confidence: {r.confidence}",
        f"Date: {r.created_at.strftime('%Y-%m-%d %H:%M')}", "",
        "Note: This report is for preliminary awareness and educational purposes only."
    ]
    for line in lines:
        pdf.drawString(50, y, str(line)[:95])
        y -= 25
    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="medical_prediction_report.pdf", mimetype="application/pdf")

def chatbot_reply(user_message, current_symptoms):
    msg = user_message.lower().strip()

    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    thanks = ["thank you", "thanks", "ok", "okay"]
    unclear = ["not feeling well", "something wrong", "i am sick", "i feel sick", "not well", "unwell"]

    if any(word in msg for word in greetings):
        return "Hello! I’m here to help you. Please tell me what problem you are facing today."

    if any(word in msg for word in thanks):
        return "You are welcome. Please take care of yourself, and consult a doctor if symptoms continue."

    if any(word in msg for word in unclear):
        return (
            "I’m sorry you are not feeling well. Can you tell me your main symptoms? "
            "For example: fever, cough, headache, stomach pain, chest pain, weakness, vomiting, or body pain."
        )

    if "pain" in msg and len(current_symptoms) == 0:
        return (
            "I understand you have pain. Where exactly is the pain? "
            "Headache, chest pain, stomach pain, back pain, or joint pain?"
        )

    if len(current_symptoms) == 0:
        return (
            "I could not identify clear symptoms yet. Please describe what you are feeling, "
            "such as fever, cough, headache, nausea, weakness, or breathing difficulty."
        )

    if len(current_symptoms) == 1:
        return (
            "I detected "
            + current_symptoms[0].replace("_", " ").title()
            + ". Please tell me 2 or 3 more symptoms so I can understand your condition better."
        )

    if len(current_symptoms) == 2:
        return (
            "I detected "
            + ", ".join([s.replace("_", " ").title() for s in current_symptoms])
            + ". Please tell me one more symptom for a better prediction."
        )

    return None


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "chat_symptoms" not in session:
        session["chat_symptoms"] = []
        session["chat_messages"] = [
            {
                "sender": "bot",
                "message": "Hello, I am your AI medical assistant. Please tell me how you are feeling today."
            }
        ]

    if request.method == "POST":
        user_message = request.form.get("message", "").strip()

        if user_message:
            session["chat_messages"].append({
                "sender": "user",
                "message": user_message
            })

            detected_symptoms = extract_symptoms_ai(user_message)
            current_symptoms = session.get("chat_symptoms", [])

            for symptom in detected_symptoms:
                if symptom not in current_symptoms:
                    current_symptoms.append(symptom)

            session["chat_symptoms"] = current_symptoms

            normal_reply = chatbot_reply(user_message, current_symptoms)

            if normal_reply:
                bot_reply = normal_reply
                if len(current_symptoms) < 4:
                    bot_reply = chatbot_reply(user_message, current_symptoms)
            else:

                values = [
                    1 if symptom in current_symptoms else 0
                    for symptom in FEATURES
                ]

                probabilities = model.predict_proba([values])[0]
                top3_indices = probabilities.argsort()[-3:][::-1]
                top3 = [(model.classes_[i], round(probabilities[i] * 100, 2)) for i in top3_indices]

                prediction = top3[0][0]
                confidence = top3[0][1]

                info = DISEASE_INFO.get(prediction, {
                    "description": "Detailed information is not available for this disease.",
                    "doctor": "General Physician"
                })

                bot_reply = (
                    f"Based on the symptoms I detected, the possible condition may be {prediction}. "
                    f"The confidence level is {confidence}%. "
                    f"{info['description']} "
                    f"Please consult a {info['doctor']} for proper medical confirmation."
                )

            session["chat_messages"].append({
                "sender": "bot",
                "message": bot_reply
            })

            session.modified = True

    return render_template(
        "chat.html",
        messages=session.get("chat_messages", []),
        detected_symptoms=[
            s.replace("_", " ").title()
            for s in session.get("chat_symptoms", [])
        ]
    )

@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_symptoms", None)
    session.pop("chat_messages", None)
    return redirect(url_for("chat"))

@app.route("/appointment", methods=["GET", "POST"])
def appointment():
    if request.method == "POST":
        patient_name = request.form.get("patient_name")
        phone = request.form.get("phone")
        doctor = request.form.get("doctor")
        appointment_date = request.form.get("appointment_date")
        appointment_time = request.form.get("appointment_time")
        reason = request.form.get("reason")

        new_appointment = Appointment(
            patient_name=patient_name,
            phone=phone,
            doctor=doctor,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            reason=reason
        )

        db.session.add(new_appointment)
        db.session.commit()

        flash("Appointment booked successfully.")
        return redirect(url_for("appointment"))

    return render_template("appointment.html")
@app.route("/appointments")
def appointments():
    all_appointments = Appointment.query.order_by(Appointment.created_at.desc()).all()
    return render_template("appointments.html", appointments=all_appointments)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (
    request.form.get("username") == os.getenv("ADMIN_USERNAME")
    and request.form.get("password") == os.getenv("ADMIN_PASSWORD")
):
            session["admin"] = True
            return redirect(url_for("history"))
        flash("Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
