import os
import io
import datetime
import pickle
from functools import wraps
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, make_response,
    redirect, url_for, session, flash, send_from_directory
)
from google import genai
from google.genai import types

# Local DB & Validation modules
from db import init_db, create_user, get_user_by_email, get_user_by_id, verify_user
from schemas import (
    UserRegisterSchema, UserLoginSchema,
    DiagnosisFormSchema, MedicineQuerySchema, ChatQuerySchema,
    validate_form
)

# Load environment configuration
# Load .env from the same folder as this app.py file
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").replace("models/", "")
DEVELOPER_NAME = "Sadhana Patel"
DEVELOPER_LINKEDIN = "https://www.linkedin.com/in/sadhana-patel-9baab82b0"

# Initialize SQLite database
init_db()

# Initialize Gemini AI client with low temperature to minimize hallucinations
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Create a .env file in the same folder as app.py "
        "and add: GOOGLE_API_KEY=YOUR_NEW_GEMINI_API_KEY"
    )

client = genai.Client(api_key=GOOGLE_API_KEY)
generation_config = types.GenerateContentConfig(
    temperature=0.15,
    top_p=0.85,
    top_k=40,
    max_output_tokens=2048,
)

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'hospital_clinical_workstation_secret_2026')
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

# --- Ensure Directories for Saved Interactions ---
interactions_dir_diagnosis = "diagnosis_database"
interactions_dir_medicine = "medicine_database"
interactions_dir_image_analysis = "image_analysis"
interactions_dir_chat_history = "chat_history"

# --- Machine Learning Disease Prediction ---
ML_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "disease_model.pkl")
ML_ARTIFACT = None
ML_MODEL_ERROR = None


def load_ml_model():
    """Load the trained Random Forest artifact lazily so Flask can still start before training."""
    global ML_ARTIFACT, ML_MODEL_ERROR
    if ML_ARTIFACT is not None:
        return ML_ARTIFACT
    try:
        with open(ML_MODEL_PATH, "rb") as model_file:
            artifact = pickle.load(model_file)
        if not isinstance(artifact, dict) or "model" not in artifact or "feature_names" not in artifact:
            raise ValueError("Invalid disease model artifact.")
        if len(artifact["feature_names"]) != 132:
            raise ValueError("Disease model metadata must contain exactly 132 symptom features.")
        ML_ARTIFACT = artifact
        ML_MODEL_ERROR = None
        return artifact
    except Exception as exc:
        ML_MODEL_ERROR = str(exc)
        return None


def predict_disease(selected_symptoms):
    """Convert selected Kaggle symptom names to the exact training feature order and predict."""
    artifact = load_ml_model()
    if artifact is None:
        raise RuntimeError(
            "The ML disease model could not be loaded. Please run 'python train_model.py' first. "
            f"Details: {ML_MODEL_ERROR or 'unknown model-loading error'}"
        )

    feature_names = artifact["feature_names"]
    allowed = set(feature_names)
    selected = set(selected_symptoms)
    invalid = selected - allowed
    if invalid:
        raise ValueError("One or more selected symptoms are invalid.")
    if not selected:
        raise ValueError("Please select at least one symptom.")

    # Build a single 0/1 row in exactly the feature order used during training.
    feature_row = pd.DataFrame(
        [[1 if feature in selected else 0 for feature in feature_names]],
        columns=feature_names
    )
    prediction = artifact["model"].predict(feature_row)[0]
    return str(prediction), len(selected), feature_names


for d in [interactions_dir_diagnosis, interactions_dir_medicine, interactions_dir_image_analysis, interactions_dir_chat_history]:
    os.makedirs(d, exist_ok=True)

# --- Context Processor for Global Template Variables ---
@app.context_processor
def inject_globals():
    user_id = session.get('user_id')
    user_data = get_user_by_id(user_id) if user_id else None
    return {
        'developer_name': DEVELOPER_NAME,
        'developer_linkedin': DEVELOPER_LINKEDIN,
        'gemini_model_name': GEMINI_MODEL.replace('models/', ''),
        'current_user': user_data
    }

# --- Authentication Helpers & Decorator ---
def get_current_user_id() -> str:
    """Returns the current logged in user's string ID for isolating history folders."""
    uid = session.get('user_id')
    return f"user_{uid}" if uid else None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access the Hospital Clinical Workstation.", "warning")
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function

# --- Interaction History Storage ---
def save_interaction(user_input: str, insights: str, directory: str):
    """Saves user interaction safely to the user's isolated directory."""
    user_dir_id = get_current_user_id()
    if not user_dir_id:
        return
    
    user_dir = os.path.join(directory, user_dir_id)
    os.makedirs(user_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(user_dir, f"interaction_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 40 + "\n")
        f.write("Clinical Input / Inquiry:\n")
        f.write(user_input + "\n\n")
        f.write("Clinical Evaluation & Recommendations:\n")
        f.write(insights)

def read_interactions_from_file(directory: str):
    """Reads saved interactions for the logged in user."""
    user_dir_id = get_current_user_id()
    if not user_dir_id:
        return []
    
    user_dir = os.path.join(directory, user_dir_id)
    if not os.path.exists(user_dir):
        return []
    
    interactions = []
    for filename in sorted(os.listdir(user_dir), reverse=True):
        if filename.endswith(".txt"):
            filepath = os.path.join(user_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                interactions.append({"filename": filename, "content": content})
    return interactions

# --- Strictly Grounded Clinical AI Core Functions (Low Hallucination) ---
def get_gemini_response(prompt: str) -> str:
    """Invokes Gemini 2.5 Flash for clinical text generation with low hallucination configuration."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=generation_config,
        )
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with clinical AI engine: {str(e)}"

def get_medical_insights(symptoms=None, disease=None, gender="Unspecified", age="N/A",
                         diabetes="no", previous_diseases="none") -> str:
    """Generates structured, clinically grounded medical insights based on symptoms or confirmed conditions."""
    if symptoms:
        prompt = (
            "You are a clinical decision support system for hospital physicians.\n"
            "Ground your analysis strictly in verified medical science and established clinical practice guidelines "
            "(e.g., UpToDate, Harrison's Principles of Internal Medicine, CDC, WHO, ACC/AHA).\n"
            "Do NOT speculate, guess, or hallucinate rare pathologies when standard presentations apply.\n\n"
            f"PATIENT CONTEXT: Biological Sex: {gender} | Age: {age} | Diabetes: {diabetes} | Medical History: {previous_diseases}.\n"
            f"CHIEF COMPLAINT / SYMPTOMS: {symptoms}\n\n"
            "Provide a concise, highly structured clinical evaluation with these exact sections:\n\n"
            "### Clinical Assessment\n"
            "Objective breakdown of reported symptoms and pathophysiology.\n\n"
            "### Differential Diagnosis\n"
            "List most probable conditions based on clinical likelihood, noting primary supporting factors.\n\n"
            "### Recommended Diagnostic Workup\n"
            "Specific initial diagnostic tests (e.g. CBC, BMP, ECG, Chest X-ray, specialized assays) to rule in/out diagnoses.\n\n"
            "### Immediate Clinical Precautions & Management\n"
            "Key acute precautions, red flag warning signs requiring immediate emergency evaluation, and initial supportive care.\n\n"
            "### Potential Pharmacotherapy Options\n"
            "Standard first-line medication classes (with standard cautions and contraindication warnings).\n\n"
            "### Lifestyle & Follow-Up\n"
            "Evidence-based dietary advice, activity modifications, and recommended physician follow-up timeframe."
        )
    elif disease:
        prompt = (
            "You are a clinical decision support system for hospital physicians.\n"
            "Ground your analysis strictly in verified medical science and established clinical practice guidelines.\n"
            "Do NOT hallucinate, guess, or provide unverified treatments.\n\n"
            f"PATIENT CONTEXT: Biological Sex: {gender} | Age: {age} | Diabetes: {diabetes} | Medical History: {previous_diseases}.\n"
            f"CONFIRMED CONDITION / INQUIRY: {disease}\n\n"
            "Provide a structured, authoritative clinical summary with these exact sections:\n\n"
            "### Pathophysiology & Clinical Presentation\n"
            "Concise description of etiology, disease progression, and cardinal symptoms.\n\n"
            "### Standard Diagnostic Workup & Monitoring\n"
            "Core laboratory markers, imaging modalities, and objective monitoring parameters.\n\n"
            "### Evidence-Based Pharmacotherapy\n"
            "First-line and second-line therapeutic agents, typical mechanisms, and key contraindications.\n\n"
            "### Inpatient / Outpatient Management Protocols\n"
            "Clinical care pathways, complication prevention, and patient stabilization steps.\n\n"
            "### Dietary & Activity Guidelines\n"
            "Scientifically supported dietary restrictions/recommendations and physical activity guidance."
        )
    else:
        return "Please specify either patient symptoms or a disease condition."

    return get_gemini_response(prompt)

def get_medicine_info(name: str, search_by: str = "generic", country: str = "US") -> str:
    """Retrieves authoritative pharmaceutical monograph details with zero speculation."""
    prompt = (
        "You are an inpatient clinical pharmacy decision support system.\n"
        "Ground all information strictly in official pharmacopeial monographs (FDA, EMA, British National Formulary).\n"
        "Do NOT invent dosages, unapproved indications, or unverified drug interactions.\n\n"
        f"MEDICATION QUERY: '{name}' (Classification: {search_by.upper()} Name, Jurisdiction: {country.upper()})\n\n"
        "Provide a comprehensive, authoritative monograph with these exact sections:\n\n"
        "### Therapeutic Indications\n"
        "FDA/EMA approved indications and common evidence-based hospital uses.\n\n"
        "### Mechanism of Action & Pharmacokinetics\n"
        "Biological mechanism, absorption, metabolism (CYP pathways), and elimination route.\n\n"
        "### Standard Adult & Pediatric Dosing\n"
        "Standard therapeutic dosing ranges, renal adjustment guidance (eGFR/CrCl), and hepatic dosing cautions.\n\n"
        "### Critical Contraindications & Drug-Drug Interactions\n"
        "Absolute contraindications and major hazardous drug interactions.\n\n"
        "### Adverse Effects Profile\n"
        "Common manageable side effects versus critical black-box warnings.\n\n"
        "### Pregnancy, Lactation & Special Populations\n"
        "Pregnancy category/lactation risk and geriatric monitoring considerations.\n\n"
        "### Brand Equivalents & Handling\n"
        "Common global brand names, standard storage requirements, and hospital administration notes."
    )
    return get_gemini_response(prompt)

def process_image(image_data: bytes, analysis_type: str) -> str:
    """Processes uploaded medical images using Gemini Vision with strict hallucination controls."""
    try:
        image = Image.open(io.BytesIO(image_data))
        image_buffer = io.BytesIO()
        image_format = image.format or 'PNG'
        image.save(image_buffer, format=image_format)
        mime_type = f"image/{image_format.lower()}" if image_format.lower() in {"png", "jpeg", "jpg", "webp", "gif"} else "image/png"

        system_grounding = (
            "You are an expert diagnostic radiologist and clinical imaging specialist assisting a hospital team.\n"
            "CRITICAL INSTRUCTIONS TO PREVENT HALLUCINATION:\n"
            "1. First, assess the validity of the image. If the image is non-clinical, an artistic illustration, simulated graphic, "
            "or of insufficient diagnostic quality, state clearly in ONE sentence: 'Notice: This image appears to be a non-clinical graphic or artistic representation rather than a true diagnostic radiograph.'\n"
            "2. Do NOT invent fractures, infiltrates, tumors, or pathology that are not clearly demonstrable.\n"
            "3. State only objective, observable anatomical structures and radiological density variations.\n"
            "4. Clearly distinguish between normal anatomy and actual suspicious lesions.\n\n"
        )

        prompts = {
            "xray_description": (
                system_grounding +
                "Analyze this radiographic image and provide findings under these exact headings:\n\n"
                "### Technical Description\n"
                "- **Projection & Anatomy:** View angle, anatomical region identified, patient positioning.\n"
                "- **Exposure Quality:** Diagnostic clarity, contrast, and visible bony/soft tissue margins.\n\n"
                "### Objective Radiological Findings\n"
                "- **Skeletal Framework:** Cortical continuity, joint alignment, and presence or absence of fracture/dislocation.\n"
                "- **Soft Tissues & Parenchyma:** Surrounding soft tissue density and presence or absence of swelling or abnormal opacities.\n\n"
                "### Clinical Impression & Recommendations\n"
                "- **Diagnostic Impression:** Summary of findings (or confirmation of non-diagnostic image).\n"
                "- **Recommended Clinical Next Steps:** Appropriate confirmatory standard multi-view projections if clinically indicated."
            ),
            "skin_cancer": (
                system_grounding +
                "Analyze this dermatological skin lesion and provide findings under these exact headings:\n\n"
                "### Visual Morphological Assessment\n"
                "- **ABCDE Evaluation:** Asymmetry, Border irregularity, Color variegation, approximate Diameter, and Evolution cues.\n\n"
                "### Differential Considerations\n"
                "- Potential benign vs atypical conditions based strictly on visible dermoscopic features.\n\n"
                "### Clinical Recommendation\n"
                "- Urgency of in-person dermatoscopic evaluation, biopsy requirement, or routine monitoring."
            ),
            "tumor_detection": (
                system_grounding +
                "Analyze this scan for suspicious masses under these exact headings:\n\n"
                "### Anatomical Localization\n"
                "- Body region, tissue plane, and spatial orientation.\n\n"
                "### Lesion Characteristics\n"
                "- Margin definition, internal density/echogenicity, and mass effect.\n\n"
                "### Recommended Diagnostic Protocol\n"
                "- Necessary contrast-enhanced imaging, MRI, or tissue biopsy protocols."
            ),
            "pregnancy_detection": (
                system_grounding +
                "Analyze this ultrasound study under these exact headings:\n\n"
                "### Sonographic Observations\n"
                "- Identifiable pelvic/uterine structures, gestational sac, or fetal pole indicators.\n\n"
                "### Clinical Interpretation\n"
                "- Objective assessment of observable sonographic landmarks.\n\n"
                "### Next Steps\n"
                "- Recommended serum beta-hCG confirmation and formal OB/GYN ultrasound scheduling."
            ),
            "lesion_detection": (
                system_grounding +
                "Analyze this wound, rash, or tissue lesion under these exact headings:\n\n"
                "### Lesion Morphology\n"
                "- Distribution, border demarcation, erythema, exudate, or central necrosis.\n\n"
                "### Etiology Hypotheses\n"
                "- Potential causes (contact dermatitis, bacterial/fungal infection, trauma, insect bite).\n\n"
                "### Wound Care Recommendations\n"
                "- Cleansing, sterile dressing, and clinical red flags for systemic infection."
            ),
            "medical_image": (
                system_grounding +
                "Analyze this clinical photograph/specimen under these exact headings:\n\n"
                "### Image Identification\n"
                "- Subject matter, anatomical location, and clinical setting.\n\n"
                "### Observable Findings\n"
                "- Visible anatomical structures and notable irregularities.\n\n"
                "### Professional Guidance\n"
                "- Summary and recommendation for healthcare provider correlation."
            )
        }

        selected_prompt = prompts.get(analysis_type, prompts["medical_image"])
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[selected_prompt, types.Part.from_bytes(data=image_buffer.getvalue(), mime_type=mime_type)],
            config=generation_config,
        )
        return response.text.strip()
    except Exception as e:
        return f"Image processing failed: {str(e)}"

def process_custom_image(image_data: bytes, custom_prompt: str) -> str:
    """Processes uploaded image using a custom prompt with strict hallucination controls."""
    try:
        image = Image.open(io.BytesIO(image_data))
        image_buffer = io.BytesIO()
        image_format = image.format or 'PNG'
        image.save(image_buffer, format=image_format)
        mime_type = f"image/{image_format.lower()}" if image_format.lower() in {"png", "jpeg", "jpg", "webp", "gif"} else "image/png"
        prompt_full = (
            "You are a hospital clinical imaging specialist assisting a physician.\n"
            "Ground your answer strictly in observable visual evidence. Do NOT hallucinate pathology.\n"
            f"PHYSICIAN INQUIRY: {custom_prompt}\n\n"
            "Provide an objective, structured, and clinically precise response."
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt_full, types.Part.from_bytes(data=image_buffer.getvalue(), mime_type=mime_type)],
            config=generation_config,
        )
        return response.text.strip()
    except Exception as e:
        return f"Custom image analysis failed: {str(e)}"

# --- Routes ---

@app.route('/')
@login_required
def index():
    """Hospital Clinical Overview Dashboard."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Hospital Staff Email Login with Pydantic Validation."""
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    error_msg = None
    form_data = {}
    
    if request.method == 'POST':
        form_data = {
            'email': request.form.get('email', '').strip(),
            'password': request.form.get('password', '')
        }
        
        validated_schema, validation_err, _ = validate_form(UserLoginSchema, form_data)
        if validation_err:
            error_msg = validation_err
        else:
            user = verify_user(validated_schema.email, validated_schema.password)
            if user:
                session.permanent = True
                session['user_id'] = user['id']
                session['user_email'] = user['email']
                session['user_name'] = user['full_name'] or user['email'].split('@')[0].capitalize()
                flash(f"Authenticated: Welcome to Hospital Workstation, Dr. {session['user_name']}.", "success")
                next_page = request.args.get('next')
                return redirect(next_page if next_page else url_for('index'))
            else:
                error_msg = "Invalid institutional credentials. Please check your email and password."

    return render_template('login.html', error=error_msg, form_data=form_data)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Hospital Staff Registration with Pydantic Validation."""
    if 'user_id' in session:
        return redirect(url_for('index'))

    error_msg = None
    form_data = {}

    if request.method == 'POST':
        form_data = {
            'full_name': request.form.get('full_name', '').strip(),
            'email': request.form.get('email', '').strip(),
            'password': request.form.get('password', ''),
            'confirm_password': request.form.get('confirm_password', '')
        }

        validated_schema, validation_err, _ = validate_form(UserRegisterSchema, form_data)
        if validation_err:
            error_msg = validation_err
        else:
            existing_user = get_user_by_email(validated_schema.email)
            if existing_user:
                error_msg = "An account with this institutional email already exists. Please log in."
            else:
                try:
                    new_user = create_user(
                        email=validated_schema.email,
                        password=validated_schema.password,
                        full_name=validated_schema.full_name
                    )
                    session.permanent = True
                    session['user_id'] = new_user['id']
                    session['user_email'] = new_user['email']
                    session['user_name'] = new_user['full_name']
                    flash("Staff account provisioned successfully. Welcome to Metropolitan General Hospital.", "success")
                    return redirect(url_for('index'))
                except Exception as e:
                    error_msg = f"Failed to provision staff account: {str(e)}"

    return render_template('signup.html', error=error_msg, form_data=form_data)

@app.route('/logout')
def logout():
    """Clears staff session and logs out."""
    session.clear()
    flash("You have successfully signed out of the hospital workstation.", "info")
    return redirect(url_for('login'))

@app.route('/diagnosis', methods=['GET', 'POST'])
@login_required
def diagnosis_index():
    """Triage & Diagnosis Module. ML is primary for selected-symptom disease prediction; Gemini remains for explanation."""
    artifact = load_ml_model()
    symptom_options = artifact["feature_names"] if artifact else []

    if request.method == 'POST':
        selected_symptoms = [
            value.strip()
            for value in request.form.getlist('symptoms')
            if value and value.strip()
        ]

        selected_text = ", ".join(selected_symptoms)
        symptom_notes = request.form.get('symptom_notes', '').strip()

        form_dict = {
            'choice': request.form.get('choice', '1'),
            'gender': request.form.get('gender', 'Male'),
            'age': request.form.get('age', '30'),
            'diabetes': request.form.get('diabetes', 'no'),
            'previous_diseases': request.form.get('previous_diseases', 'none'),
            # Use selected ML symptoms for schema validation; notes remain an optional addition.
            'symptoms': selected_text or symptom_notes,
            'disease': request.form.get('disease', '')
        }

        if form_dict['choice'] == '1' and not selected_symptoms:
            flash("Please select at least one symptom before predicting the disease.", "error")
            return render_template(
                'diagnosis_index.html',
                form_data=form_dict,
                symptom_options=symptom_options,
                selected_symptoms=selected_symptoms
            )

        validated_schema, err, _ = validate_form(DiagnosisFormSchema, form_dict)
        if err:
            flash(f"Clinical Validation Notice: {err}", "error")
            return render_template(
                'diagnosis_index.html',
                form_data=form_dict,
                symptom_options=symptom_options,
                selected_symptoms=selected_symptoms
            )

        if form_dict['choice'] == '1':
            try:
                predicted_disease, symptom_count, _ = predict_disease(selected_symptoms)
            except (RuntimeError, ValueError) as exc:
                flash(f"ML Prediction Error: {exc}", "error")
                return render_template(
                    'diagnosis_index.html',
                    form_data=form_dict,
                    symptom_options=symptom_options,
                    selected_symptoms=selected_symptoms
                )

            gemini_symptoms = selected_text
            if symptom_notes:
                gemini_symptoms += f"\nAdditional clinical notes: {symptom_notes}"

            # Gemini remains available, but it is explicitly used as an explanation layer.
            insights = get_medical_insights(
                symptoms=gemini_symptoms,
                gender=validated_schema.gender,
                age=validated_schema.age,
                diabetes=validated_schema.diabetes,
                previous_diseases=validated_schema.previous_diseases
            )
            insights = (
                f"### ML Prediction\n"
                f"**Primary Predicted Disease:** {predicted_disease}\n"
                f"**Number of Symptoms:** {symptom_count}\n"
                f"**Model:** Random Forest\n\n"
                f"### Gemini Clinical Explanation\n{insights}"
            )

            input_summary = (
                f"ML Prediction: {predicted_disease}\n"
                f"Selected Symptoms ({symptom_count}): {selected_text}\n"
                f"Gender: {validated_schema.gender}, Age: {validated_schema.age}, "
                f"Diabetes: {validated_schema.diabetes}, History: {validated_schema.previous_diseases}"
            )
            save_interaction(input_summary, insights, interactions_dir_diagnosis)
            return render_template(
                'diagnosis_results.html',
                insights=insights,
                ml_prediction=predicted_disease,
                symptom_count=symptom_count,
                ml_model_name="Random Forest",
                selected_symptoms=selected_symptoms,
                symptoms=selected_text,
                gender=validated_schema.gender,
                age=validated_schema.age,
                diabetes=validated_schema.diabetes,
                previous_diseases=validated_schema.previous_diseases
            )

        # Preserve the existing known-disease/Gemini pathway.
        insights = get_medical_insights(
            disease=validated_schema.disease,
            gender=validated_schema.gender,
            age=validated_schema.age,
            diabetes=validated_schema.diabetes,
            previous_diseases=validated_schema.previous_diseases
        )
        input_summary = f"Condition: {validated_schema.disease}, Gender: {validated_schema.gender}, Age: {validated_schema.age}, Diabetes: {validated_schema.diabetes}, History: {validated_schema.previous_diseases}"
        save_interaction(input_summary, insights, interactions_dir_diagnosis)
        return render_template(
            'diagnosis_results.html',
            insights=insights,
            disease=validated_schema.disease,
            gender=validated_schema.gender,
            age=validated_schema.age,
            diabetes=validated_schema.diabetes,
            previous_diseases=validated_schema.previous_diseases
        )

    return render_template(
        'diagnosis_index.html',
        symptom_options=symptom_options,
        selected_symptoms=[]
    )

@app.route("/medicine", methods=["GET", "POST"])
@login_required
def medicine_index():
    """Hospital Pharmacy & Formulary Module with Pydantic validation."""
    if request.method == "POST":
        form_dict = {
            'search_by': request.form.get("search_by", "generic"),
            'name': request.form.get("name", "").strip(),
            'country': request.form.get("country", "US")
        }

        validated_schema, err, _ = validate_form(MedicineQuerySchema, form_dict)
        if err:
            flash(f"Formulary Notice: {err}", "error")
            return render_template("medicine_index.html", form_data=form_dict)

        info = get_medicine_info(
            name=validated_schema.name,
            search_by=validated_schema.search_by,
            country=validated_schema.country
        )
        input_summary = f"Search By: {validated_schema.search_by}, Compound: {validated_schema.name}, Standards: {validated_schema.country}"
        save_interaction(input_summary, info, interactions_dir_medicine)
        return render_template("medicine_results.html", info=info, medicine_name=validated_schema.name, search_by=validated_schema.search_by)

    return render_template("medicine_index.html")

@app.route('/image', methods=['GET', 'POST'])
@login_required
def image_analysis():
    """PACS Radiology & Diagnostic Vision Analysis."""
    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type', 'xray_description')
        custom_prompt = request.form.get('custom_prompt', '').strip()

        if 'image' not in request.files or request.files['image'].filename == '':
            flash("Please choose a medical diagnostic scan file (JPEG/PNG/TIFF) to examine.", "error")
            return render_template('image_index.html')

        try:
            image_file = request.files['image']
            image_bytes = image_file.read()

            if analysis_type == 'custom_image':
                if not custom_prompt:
                    flash("Please provide specific clinical instructions for this custom examination.", "error")
                    return render_template('image_index.html')
                result = process_custom_image(image_bytes, custom_prompt)
            else:
                result = process_image(image_bytes, analysis_type)

            summary = f"Modality Protocol: {analysis_type}\nCustom Query: {custom_prompt or 'Standard'}"
            save_interaction(summary, result, interactions_dir_image_analysis)
            return render_template('image_results.html', analysis_result=result, analysis_type=analysis_type)
        except Exception as e:
            flash(f"PACS Interpretation Error: {str(e)}", "error")
            return render_template('image_index.html')

    return render_template('image_index.html')

@app.route('/chat', methods=["GET", "POST"])
@login_required
def chatbot():
    """Inpatient Physician Rounds & Clinical Decision Support."""
    if request.method == "POST":
        form_dict = {
            'prompt': request.form.get('prompt', '').strip(),
            'topic': request.form.get('topic', '')
        }

        validated_schema, err, _ = validate_form(ChatQuerySchema, form_dict)
        if err:
            flash(f"Clinical Notice: {err}", "error")
            return render_template("chat_index.html", form_data=form_dict)

        num_molecules = request.form.get('num-molecules', '0')
        molecules_data = []
        try:
            n_mols = int(num_molecules)
            for i in range(1, n_mols + 1):
                m_name = request.form.get(f'molecule-{i}', '').strip()
                m_prop = request.form.get(f'property-{i}', '').strip()
                if m_name:
                    molecules_data.append((m_name, m_prop))
        except ValueError:
            pass

        full_prompt = (
            "You are a senior clinical consultant providing physician decision support.\n"
            "Ground your advice strictly in verified clinical evidence and hospital protocols.\n"
        )
        if validated_schema.topic:
            full_prompt += f"Clinical Specialty: {validated_schema.topic}\n"
        if molecules_data:
            full_prompt += "Target Molecular Agents Considered:\n"
            for idx, (m_name, m_prop) in enumerate(molecules_data, 1):
                full_prompt += f"  {idx}. Agent: {m_name} | Target Property: {m_prop}\n"
        full_prompt += f"\nClinical Case Inquiry:\n{validated_schema.prompt}\n\n"
        full_prompt += "Provide an objective, structured, evidence-grounded physician consultation summary."

        response_text = get_gemini_response(full_prompt)
        save_interaction(validated_schema.prompt, response_text, interactions_dir_chat_history)
        return render_template("chat_results.html", user_input=validated_schema.prompt, response=response_text)

    return render_template("chat_index.html")

@app.route('/history')
@login_required
def history():
    """Patient Medical Records Vault."""
    diagnosis_history = read_interactions_from_file(interactions_dir_diagnosis)
    medicine_history = read_interactions_from_file(interactions_dir_medicine)
    image_history = read_interactions_from_file(interactions_dir_image_analysis)
    chatbot_history = read_interactions_from_file(interactions_dir_chat_history)

    return render_template('history.html',
                           diagnosis_history=diagnosis_history,
                           medicine_history=medicine_history,
                           image_history=image_history,
                           chatbot_history=chatbot_history)

@app.route('/history/<interaction_type>/<filename>')
@login_required
def download_interaction(interaction_type, filename):
    """Secure Chart Note File Download."""
    user_dir_id = get_current_user_id()
    if not user_dir_id:
        return "Unauthorized", 401

    dir_map = {
        'diagnosis': interactions_dir_diagnosis,
        'medicine': interactions_dir_medicine,
        'image': interactions_dir_image_analysis,
        'chat': interactions_dir_chat_history
    }
    target_dir = dir_map.get(interaction_type)
    if not target_dir:
        return "Invalid category", 404

    user_dir = os.path.join(target_dir, user_dir_id)
    filepath = os.path.join(user_dir, os.path.basename(filename))
    if not os.path.exists(filepath):
        return "File not found", 404

    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read()

    response = make_response(content)
    response.headers['Content-Type'] = 'text/plain; charset=utf-8'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
