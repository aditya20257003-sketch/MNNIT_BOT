import os
import json
import datetime
import threading
import tempfile
import time
import numpy as np
import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
import easyocr
from PIL import Image
from gtts import gTTS
import pygame
from groq import Groq  # <-- Replaced 'import ollama' for zero-install cloud portability
import ipywidgets as widgets
from IPython.display import display
import torch
import requests
from bs4 import BeautifulSoup

# --- CONFIGURATION (Ensure this runs first) ---
if 'SAVE_DIR' not in locals():
    SAVE_DIR = "chatbot_data" 
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

json_path = os.path.join(SAVE_DIR, "manual_knowledge.json")

# --- INTEGRATED SCRAPING FUNCTION ---
def scrape_and_add_knowledge(url, category):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 100]
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        if category not in data:
            data[category] = []
        
        for p in paragraphs[:5]: 
            if p not in data[category]:
                data[category].append(p)
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"Success! Added data to '{category}'."
    except Exception as e:
        return f"Error: {e}"

print("Knowledge Management tools loaded successfully.")
# --- CONFIGURATION & DIRECTORIES ---
BASE_DIR = r"C:\Users\Aditya Ranjan\Desktop\MNNIT_Chatbot"
if not os.path.exists(os.path.dirname(BASE_DIR)):
    BASE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "MNNIT_Chatbot")

SAVE_DIR = os.path.join(BASE_DIR, "chatbot_data")
CONV_DIR = os.path.join(BASE_DIR, "chatbot_conversations")
DB_DIR   = os.path.join(BASE_DIR, "chatbot_db")

for d in [SAVE_DIR, CONV_DIR, DB_DIR]:
    os.makedirs(d, exist_ok=True)

# --- USER STORAGE REGISTER ---
USERS_DB_PATH = os.path.join(DB_DIR, "user_registry.json")
if not os.path.exists(USERS_DB_PATH):
    with open(USERS_DB_PATH, "w") as f:
        json.dump({}, f)

USER_NAME = "Adi"  # Dynamically mutated upon successful authentication
ADMIN_PASSWORD = "mnnit_admin_secure"  
STATE = {"voice_mode": 0, "is_speaking": False}

stop_speech = threading.Event()
cam_stop    = threading.Event()

WORDS = ['hello','hi','hey','college','hostel','mess','campus',
         'professor','consulting','strategy','market','business',
         'case','gym','workout','protein','fitness','health',
         'yes','no','okay','thanks','sorry','help','stop',
         'good','bad','please','welcome','goodbye']

# --- GROQ API INITIALIZATION ---
GROQ_API_KEY = "API_KEY_CODE"
client = Groq(api_key=GROQ_API_KEY)

# --- KNOWLEDGE BASE MANAGEMENT ---
json_path = os.path.join(SAVE_DIR, "manual_knowledge.json")
default_kb = {
    "nit_allahabad": [
        "MNNIT Allahabad is a National Institute of Technology in Prayagraj",
        "First year students stay in Himalaya or Vindhya hostel",
        "The mess serves breakfast lunch and dinner daily",
        "MNNIT has departments including Mechanical CSE ECE Civil Chemical",
        "The campus has a gym football ground and basketball court",
        "Placements happen in final year through Training and Placement cell",
        "MNNIT was established in 1961 as Motilal Nehru Regional Engineering College",
        "The director of MNNIT is a senior professor appointed by MHRD",
        "MNNIT has around 5000 students across UG PG and PhD programs"
    ],
    "consultancy": [
        "Case interviews require structured frameworks like MECE",
        "McKinsey BCG and Bain are top tier consulting firms",
        "Market entry cases need analysis of market size competition and barriers",
        "Profitability cases split into revenue side and cost side",
        "Porter Five Forces covers rivalry suppliers buyers substitutes new entrants",
        "Always structure answers with Situation Complication Resolution",
        "Guesstimate cases require logical breakdown of numbers"
    ],
    "health_gym": [
        "Protein intake should be 1.6 to 2.2 grams per kg of bodyweight",
        "Compound exercises like squat bench deadlift build the most muscle",
        "Sleep 7 to 9 hours for optimal muscle recovery",
        "Progressive overload is the key principle for strength gains",
        "Caloric surplus of 200 to 300 calories needed for lean bulking",
        "Creatine monohydrate 5g daily is the most proven supplement",
        "Deload every 6 to 8 weeks to avoid overtraining"
    ]
}

if not os.path.exists(json_path):
    with open(json_path, "w") as f:
        json.dump(default_kb, f, indent=2)

with open(json_path, "r") as f:
    manual_knowledge = json.load(f)

# --- INITIALIZATION SEQUENCE ---
pygame.mixer.init()
print("System core initialization sequence initiated...")
print("Hardware acceleration (CUDA):", "ENABLED" if torch.cuda.is_available() else "DISABLED")
if torch.cuda.is_available():
    print("Active Compute Device:", torch.cuda.get_device_name(0))

print("Loading EasyOCR engine models...")
ocr_reader = easyocr.Reader(['en', 'hi'], gpu=torch.cuda.is_available())
print("All runtime core systems online.")

# --- CASCADE CLASSIFIERS ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# --- CHATBOT CORE ENGINE ---
conversation_history = []
conversation_log     = []

BADGE = {
    'nit_allahabad': 'MNNIT CAMPUS',
    'consultancy':   'STRATEGY CONSULTING',
    'health_gym':    'HEALTH & FITNESS',
    'general':       'GENERAL INTELLIGENCE'
}

COLORS = {
    'nit_allahabad': '#881337', 
    'consultancy':   '#7f1d1d', 
    'health_gym':    '#4c0519', 
    'general':       '#52525b'  
}

def detect_domain(q):
    q = q.lower()
    n = sum(1 for k in ['nit','mnnit','allahabad','professor','hostel','mess','campus','college','exam','placement','department','fest','senior','junior'] if k in q)
    c = sum(1 for k in ['consulting','case','mckinsey','bcg','strategy','framework','market','business','mece','porter','revenue','profit'] if k in q)
    h = sum(1 for k in ['gym','workout','fitness','protein','diet','exercise','muscle','health','calories','sleep','supplement','creatine','bulk'] if k in q)
    scores = {'nit_allahabad': n, 'consultancy': c, 'health_gym': h}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'general'

def get_sys(domain):
    return {
        'nit_allahabad': "You are an expert assistant for MNNIT Allahabad. Provide details regarding hostels, mess rules, departments, and placements based explicitly on facts.",
        'consultancy': "You are an elite management consulting expert. Use MECE principles and strategic, analytical frameworks to offer business structures.",
        'health_gym': "You are a certified fitness coach and clinical nutritionist. Provide precise, evidence-based recommendations on training and physical health.",
        'general': "You are a helpful expert assistant specialized across engineering institutions, strategy consulting, and sports science."
    }.get(domain, "You are a helpful assistant.")

def speak(text):
    if not text: return
    stop_speech.clear()
    STATE["is_speaking"] = True
    def _play():
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            lang = 'hi' if STATE["voice_mode"] == 1 else 'en'
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(tmp.name)
            tmp.close()
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if stop_speech.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.1)
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            os.unlink(tmp.name)
        except Exception as e:
            print(f"Audio interface output fault: {e}")
        finally:
            STATE["is_speaking"] = False
    threading.Thread(target=_play, daemon=True).start()

# Modified default tracking execution to capture dynamic username mutations smoothly
def chat(question, uname=None):
    global USER_NAME
    if uname is None:
        uname = USER_NAME
        
    domain = detect_domain(question)
    facts = []
    for v in manual_knowledge.values():
        facts.extend(v)
    ctx = "\n- " + "\n- ".join(facts)
    hist = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in conversation_history[-6:])
    
    prompt = (
        f"KNOWLEDGE BASE FACTS:\n{ctx}\n\n"
        f"CONVERSATION HISTORY:\n{hist}\n\n"
        f"USER ({uname}): {question}\n\n"
        f"Answer explicitly using context data. Limit responses strictly to under 80 words for concise text-to-speech readability.")
    
    conversation_history.append({'role': 'user', 'content': question})
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {'role': 'system', 'content': get_sys(domain)},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Cloud framework endpoint transmission fault: {e}. Check API key balance parameters or upstream availability states."
        
    conversation_history.append({'role': 'assistant', 'content': answer})
    conversation_log.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'domain': domain,
        'question': question,
        'answer': answer
    })
    try:
        with open(os.path.join(CONV_DIR, f'conv_{datetime.date.today()}.json'), 'w') as f:
            json.dump(conversation_log, f, indent=2)
    except:
        pass
    return answer, domain

# --- VISION AND SPEECH FUNCTIONS ---
def get_mouth_box(frame, pad=15):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
    if len(faces) == 0: return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    fx, fy, fw, fh = faces[0]
    ly = fy + int(fh * 0.60)
    fl = gray[ly:fy+fh, fx:fx+fw]
    ms = mouth_cascade.detectMultiScale(fl, 1.5, 15, minSize=(25,15))
    if len(ms) == 0:
        return (max(0, fx+fw//4), max(0, fy+int(fh*0.68)), min(w, fx+3*fw//4), min(h, fy+fh))
    ms = sorted(ms, key=lambda m: m[2], reverse=True)
    mx2, my2, mw2, mh2 = ms[0]
    return (max(0, fx+mx2-pad), max(0, ly+my2-pad), min(w, fx+mx2+mw2+pad), min(h, ly+my2+mh2+pad))

def analyse_lips(frames, boxes):
    if len(frames) < 5: return None
    motions, prev = [], None
    for frame, box in zip(frames, boxes):
        if box is None: continue
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        g = cv2.cvtColor(cv2.resize(crop, (64, 32)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            motions.append(np.mean(cv2.absdiff(g, prev)))
        prev = g
    if not motions: return None
    avg, mx, dur = np.mean(motions), np.max(motions), sum(1 for m in motions if m > np.mean(motions)*1.2)
    score = avg*0.5 + mx*0.3 + dur*0.2
    
    short = ['hi','hey','yes','no','okay','bad','good','stop','help']
    medium = ['hello','thanks','sorry','please','welcome','goodbye','college','hostel','mess','campus','gym']
    long_w = ['professor','consulting','strategy','business','workout','protein','fitness','placement']
    
    cands = short if score < 4 else (medium if score < 10 else long_w)
    valid = [w for w in cands if w in WORDS] or WORDS
    return valid[int(dur) % len(valid)]

def frame_to_bytes(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    pil.save(buf.name, quality=75)
    buf.close()
    with open(buf.name, 'rb') as f: data = f.read()
    try: os.unlink(buf.name)
    except: pass
    return data

def open_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()
    return None

def stream_cam(img_widget, seconds=4):
    cap = open_camera()
    if cap is None: return None, None, None
    frames, boxes, last = [], [], None
    start = time.time()
    while not cam_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        box = get_mouth_box(frame)
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (136, 19, 55), 2) 
        rem = max(0, seconds - (time.time() - start))
        cv2.rectangle(frame, (0, 0), (640, 35), (24, 24, 27), -1)
        cv2.putText(frame, f"VIDEO STREAM ACCELERATION: {rem:.1f}s ACTIVE", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (244, 244, 245), 1)
        frames.append(frame.copy())
        boxes.append(box)
        last = frame.copy()
        try: img_widget.value = frame_to_bytes(frame)
        except: pass
        time.sleep(0.03)
        if time.time() - start >= seconds: break
    cap.release()
    return frames, boxes, last

def listen_mic():
    try:
        samplerate, duration = 16000, 6
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav.write(tmp.name, samplerate, recording)
        tmp.close()
        r = sr.Recognizer()
        with sr.AudioFile(tmp.name) as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.record(source)
        try: os.unlink(tmp.name)
        except: pass
        lang = "hi-IN" if STATE["voice_mode"] == 1 else "en-IN"
        return r.recognize_google(audio, language=lang).strip()
    except: return ""

# --- STYLED UI COMPONENTS ---
def st(msg, color="#a1a1aa"):
    return f"<div style='color:#f4f4f5; font-family: system-ui, sans-serif; font-size:12px; font-weight:500; border-left: 3px solid #881337; padding: 2px 0 2px 10px;'>System Log Status: <span style='color:{color}; font-weight:600; letter-spacing:0.3px;'>{msg.upper()}</span></div>"

def build_chat_html():
    if not conversation_history:
        return "<div style='color:#71717a; padding:60px 40px; text-align:center; font-family:system-ui, sans-serif; font-size:13px; letter-spacing:0.5px;'>WORKSPACE EMPTY — START SYSTEM INTERACTION TRANSACTION ABOVE</div>"
    html = ""
    pairs, temp_q = [], None
    for msg in conversation_history:
        if msg['role'] == 'user': temp_q = msg['content']
        else: pairs.append((temp_q, msg['content']))
        
    for q, a in pairs[-6:]:
        domain = detect_domain(q)
        badge_text = BADGE.get(domain, "GENERAL INTELLIGENCE")
        
        html += (
            f"<div style='margin:14px 0; text-align:right; font-family:system-ui, sans-serif;'>"
            f"<div style='background:#18181b; color:#f4f4f5; padding:10px 14px; border: 1px solid #27272a; border-right: 3px solid #881337; display:inline-block; max-width:80%; font-size:13.5px; text-align:left; border-radius: 2px;'>"
            f"<div style='color:#a1a1aa; font-size:10px; font-weight:700; margin-bottom:4px; letter-spacing:0.5px;'>OPERATOR REQUEST</div>"
            f"{q}"
            f"</div>"
            f"</div>"
        )
        html += (
            f"<div style='margin:14px 0; text-align:left; font-family:system-ui, sans-serif;'>"
            f"<div style='display:inline-block; max-width:85%; background:#111113; border: 1px solid #27272a; border-left: 4px solid #881337; padding:12px 16px; border-radius:2px;'>"
            f"<div style='color:#f43f5e; font-size:10px; font-weight:700; text-transform:uppercase; margin-bottom:6px; letter-spacing:0.8px;'>SYSTEM CORE // {badge_text}</div>"
            f"<div style='color:#e4e4e7; font-size:13.5px; line-height:1.6; font-weight:400;'>{a}</div>"
            f"</div>"
            f"</div>"
        )
    return html

# --- UI CONTROLS ---
chat_box = widgets.HTML(
    value="<div style='background:#09090b; min-height:320px; padding:20px;'></div>",
    layout=widgets.Layout(height='380px', overflow_y='auto', border='1px solid #27272a', border_radius='4px', background_color='#09090b')
)

cam_label = widgets.HTML("<div style='color:#a1a1aa; font-size:11px; font-family:system-ui, sans-serif; margin: 10px 0 4px 2px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase;'>OPTICAL FEED MONITOR</div>", layout=widgets.Layout(display='none'))
cam_img = widgets.Image(format='jpeg', width=400, layout=widgets.Layout(display='none', border='1px solid #881337', border_radius='2px', margin='0 0 12px 0'))

text_input = widgets.Text(placeholder='Input control arguments or open text dialogue parameters...', layout=widgets.Layout(width='82%', height='36px'))
send_btn = widgets.Button(description='EXECUTE', icon='paper-plane', button_style='primary', layout=widgets.Layout(width='16%', height='36px'))

mic_btn = widgets.Button(description='VOICE INPUT', button_style='info', icon='microphone', layout=widgets.Layout(width='154px', height='34px'))
ocr_btn = widgets.Button(description='OCR SCAN', button_style='warning', icon='camera', layout=widgets.Layout(width='154px', height='34px'))
lip_btn = widgets.Button(description='BIOMETRIC LIP', button_style='success', icon='eye', layout=widgets.Layout(width='154px', height='34px'))
stop_btn = widgets.Button(description='HALT AUDIO', button_style='danger', icon='stop', layout=widgets.Layout(width='154px', height='34px'))
clear_btn = widgets.Button(description='WIPE BUFFER', icon='trash', layout=widgets.Layout(width='154px', height='34px'))

status_bar = widgets.HTML(st("System structural grid online. Operations normal.", "#10b981"))
log_out = widgets.Output(layout=widgets.Layout(max_height='60px', overflow_y='auto'))

# --- ADMIN PANEL FOR EDITING ---
admin_title = widgets.HTML("<div style='color:#fafafa; font-weight:700; font-size:13px; font-family:system-ui, sans-serif; letter-spacing:0.5px; text-transform:uppercase; border-bottom: 1px solid #3f3f46; padding-bottom: 6px; margin-bottom:10px;'>DATABASE KNOWLEDGE INJECTION PANEL</div>")
pass_input = widgets.Password(placeholder='Enter encryption clearance credential key...', layout=widgets.Layout(width='260px'))
login_btn = widgets.Button(description='AUTHORIZE', icon='lock', button_style='primary', layout=widgets.Layout(width='120px'))

fact_category = widgets.Dropdown(options=[('MNNIT Campus Architecture', 'nit_allahabad'), ('Corporate Advisory Strategy', 'consultancy'), ('Biokinetics & Dietetics', 'health_gym')], description='TARGET MAP:', layout=widgets.Layout(width='280px'))
fact_input = widgets.Text(placeholder='Type factual statement node to structurally bind into JSON storage map...', layout=widgets.Layout(width='440px'))
commit_btn = widgets.Button(description='COMMIT NODE', button_style='success', icon='save', layout=widgets.Layout(width='140px'))

admin_edit_box = widgets.HBox([fact_category, fact_input, commit_btn], layout=widgets.Layout(margin='10px 0', display='none', justify_content='space-between'))
admin_panel = widgets.VBox([admin_title, widgets.HBox([pass_input, login_btn], layout=widgets.Layout(gap='8px')), admin_edit_box], layout=widgets.Layout(padding='16px', border='1px solid #27272a', border_radius='4px', margin='24px 0 0 0', background_color='#141417'))

# --- LOGIC ACTIONS ---
def refresh_chat():
    chat_box.value = f"<div style='background:#09090b; min-height:350px; padding:20px;'>{build_chat_html()}</div>"

def toggle_cam(show=True):
    state = 'block' if show else 'none'
    cam_img.layout.display = state
    cam_label.layout.display = state

def toggle_controls(activate=True):
    for button in [send_btn, mic_btn, ocr_btn, lip_btn, clear_btn]:
        button.disabled = not activate

def on_send(b=None):
    query = text_input.value.strip()
    if not query: return
    text_input.value = ""
    toggle_controls(False)
    status_bar.value = st("Routing matrix token parameters to cloud LLM framework...", "#f59e0b")
    
    def run():
        ans, domain = chat(query)
        refresh_chat()
        status_bar.value = st("Context evaluation finished. Building audio track...", "#3b82f6")
        speak(ans)
        toggle_controls(True)
        status_bar.value = st("Core operational array ready.", "#10b981")
    threading.Thread(target=run, daemon=True).start()

def on_mic(b):
    toggle_controls(False)
    status_bar.value = st("Acoustic device channel open. Awaiting user speech stream...", "#3b82f6")
    
    def run():
        captured_text = listen_mic()
        if captured_text:
            status_bar.value = st(f"Decoded trace stream: '{captured_text[:35]}...' Computing...", "#eab308")
            ans, domain = chat(captured_text)
            refresh_chat()
            status_bar.value = st("Synthesizing voice audio transmission matrix...", "#3b82f6")
            speak(ans)
            status_bar.value = st("Core operational array ready.", "#10b981")
        else:
            status_bar.value = st("Capture sequence failed: Amplitude below threshold limits.", "#ef4444")
        toggle_controls(True)
    threading.Thread(target=run, daemon=True).start()

def on_ocr(b):
    toggle_controls(False)
    toggle_cam(True)
    cam_stop.clear()
    status_bar.value = st("Optical camera thread live. Present document surface to lens...", "#eab308")
    
    def run():
        _, _, frame = stream_cam(cam_img, seconds=4)
        if frame is None:
            status_bar.value = st("System interface fault: Peripheral video capture node offline.", "#ef4444")
            toggle_cam(False); toggle_controls(True); return
        status_bar.value = st("Running deep segmentation parsing algorithms...", "#eab308")
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            segments = ocr_reader.readtext(rgb, detail=0, paragraph=True)
            parsed_text = " ".join(segments).strip()
        except Exception as e:
            parsed_text = ""
            with log_out: print(f"OCR Exception Event: {e}")
            
        if parsed_text:
            status_bar.value = st("Target text identified. Mapping vector embedding indexes...", "#3b82f6")
            ans, _ = chat(f"Explain or summarize this captured document context: {parsed_text[:450]}")
            refresh_chat()
            status_bar.value = st("Synthesizing output speech payload data...", "#3b82f6")
            speak(ans)
            status_bar.value = st("Core operational array ready.", "#10b981")
        else:
            status_bar.value = st("Optical parse finished: No textual matrices isolated.", "#ef4444")
        toggle_cam(False); toggle_controls(True)
    threading.Thread(target=run, daemon=True).start()

def on_lip(b):
    toggle_controls(False)
    toggle_cam(True)
    cam_stop.clear()
    status_bar.value = st("Biometric tracking online. Align mouth pose parameters clearly...", "#3b82f6")
    
    def run():
        frames, boxes, _ = stream_cam(cam_img, seconds=4)
        if not frames:
            status_bar.value = st("System interface fault: Frame packet streaming dropped.", "#ef4444")
            toggle_cam(False); toggle_controls(True); return
        status_bar.value = st("Processing motion velocity flow maps...", "#eab308")
        detected_token = analyse_lips(frames, boxes)
        if detected_token:
            status_bar.value = st(f"Token matches control library dictionary: [{detected_token.upper()}]", "#10b981")
            ans, _ = chat(detected_token)
            refresh_chat()
            speak(ans)
            status_bar.value = st("Core operational array ready.", "#10b981")
        else:
            status_bar.value = st("Biometric capture mismatch: Motion envelope path unverified.", "#ef4444")
        toggle_cam(False); toggle_controls(True)
    threading.Thread(target=run, daemon=True).start()

def on_abort(b):
    stop_speech.set()
    cam_stop.set()
    status_bar.value = st("Active operational loop tracks forced to safe termination state.", "#ef4444")

def on_clear(b):
    global conversation_history
    conversation_history = []
    refresh_chat()
    status_bar.value = st("Volatile session conversation stack arrays wiped clean.", "#a1a1aa")

def verify_admin(b):
    if pass_input.value == ADMIN_PASSWORD:
        status_bar.value = st("Security credential cleared. Root storage injection matrix opened.", "#10b981")
        admin_edit_box.layout.display = 'flex'
        pass_input.disabled = True
        login_btn.disabled = True
    else:
        status_bar.value = st("Security alert: Access clearance denied. Credential string invalid.", "#ef4444")
        pass_input.value = ""

def commit_fact(b):
    fact_str = fact_input.value.strip()
    target_category = fact_category.value
    if not fact_str:
        status_bar.value = st("Transaction aborted: Target assertion payload is completely null.", "#ef4444")
        return
    
    manual_knowledge[target_category].append(fact_str)
    try:
        with open(json_path, "w") as f:
            json.dump(manual_knowledge, f, indent=2)
        status_bar.value = st(f"Successfully appended structural knowledge unit under sector: {target_category.upper()}", "#10b981")
        fact_input.value = ""
    except Exception as e:
        status_bar.value = st(f"File writing blocking operation exception fault: {e}", "#ef4444")

# --- ASSIGN EVENT LISTENERS ---
send_btn.on_click(on_send)
text_input.on_submit(on_send)
mic_btn.on_click(on_mic)
ocr_btn.on_click(on_ocr)
lip_btn.on_click(on_lip)
stop_btn.on_click(on_abort)
clear_btn.on_click(on_clear)
login_btn.on_click(verify_admin)
commit_btn.on_click(commit_fact)

# --- VIEW RENDER COMPOSITION ---
ui = widgets.VBox([
    widgets.HTML(
        "<div style='color:#fafafa; font-size:18px; font-weight:700; font-family:system-ui, sans-serif; letter-spacing:0.5px; margin-bottom:14px; border-bottom: 2px solid #881337; padding-bottom: 10px; text-transform:uppercase;'>"
        "Knowledge Operations Management Console "
        "<span style='font-size:11px; color:#a1a1aa; font-weight:500; float:right; padding-top:6px; letter-spacing:1px;'>MNNIT.WORKSPACE.CORE</span>"
        "</div>"
    ),
    chat_box,
    cam_label,
    cam_img,
    widgets.HBox([text_input, send_btn], layout=widgets.Layout(justify_content='space-between', margin='14px 0 10px 0')),
    widgets.HBox([mic_btn, ocr_btn, lip_btn, stop_btn, clear_btn], layout=widgets.Layout(justify_content='space-between', margin='6px 0 16px 0')),
    status_bar,
    log_out,
    admin_panel
], layout=widgets.Layout(
    padding='24px',
    background_color='#09090b',
    border='1px solid #4c0519',
    border_radius='4px',
    width='840px'
))

# --- IDENTITY ACCESS GATEKEEPER UI (NEW FEATURE) ---
auth_title = widgets.HTML(
    "<div style='color:#fafafa; font-size:16px; font-weight:700; font-family:system-ui, sans-serif; letter-spacing:0.5px; margin-bottom:14px; border-bottom: 2px solid #881337; padding-bottom: 10px; text-transform:uppercase;'>"
    "Identity Verification Node"
    "<span style='font-size:10px; color:#a1a1aa; font-weight:500; float:right; padding-top:4px; letter-spacing:1px;'>GATEKEEPER.SECURE</span>"
    "</div>"
)

email_field = widgets.Text(placeholder='Enter operational email address...', layout=widgets.Layout(width='100%', height='36px'))
pass_field = widgets.Password(placeholder='Enter personal entry passkey...', layout=widgets.Layout(width='100%', height='36px'))
auth_status = widgets.HTML(f"<div style='color:#a1a1aa; font-family:system-ui; font-size:12px; margin-top:5px;'>Awaiting system identity initialization inputs...</div>")

submit_auth_btn = widgets.Button(description='ACCESS CORE', button_style='danger', icon='sign-in', layout=widgets.Layout(width='100%', height='38px', margin='10px 0 0 0'))

auth_card = widgets.VBox([
    auth_title,
    widgets.HTML("<span style='color:#a1a1aa; font-size:11px; font-weight:600; text-transform:uppercase;'>System Email Descriptor:</span>"),
    email_field,
    widgets.HTML("<span style='color:#a1a1aa; font-size:11px; font-weight:600; text-transform:uppercase; margin-top:8px; display:block;'>Identity Access Passkey:</span>"),
    pass_field,
    submit_auth_btn,
    auth_status
], layout=widgets.Layout(
    padding='24px',
    background_color='#09090b',
    border='1px solid #4c0519',
    border_radius='4px',
    width='400px',
    margin='40px auto'
))

# Global UI wrapper container box
root_container = widgets.VBox([auth_card], layout=widgets.Layout(width='100%', align_items='center'))

def handle_identity_verification(b):
    global USER_NAME
    email = email_field.value.strip().lower()
    password = pass_field.value.strip()
    
    if not email or "@" not in email or not password:
        auth_status.value = "<div style='color:#ef4444; font-family:system-ui; font-size:12px; margin-top:5px;'>Verification Failed: Invalid input data format structure.</div>"
        return
        
    try:
        with open(USERS_DB_PATH, "r") as f:
            registry = json.load(f)
            
        # Parse cleanly styled display name out of the provided email string structure
        extracted_handle = email.split('@')[0]
        cleaned_name = extracted_handle.replace('.', ' ').replace('_', ' ').title()
        
        if email in registry:
            if registry[email] == password:
                USER_NAME = cleaned_name
                auth_status.value = f"<div style='color:#10b981; font-family:system-ui; font-size:12px; margin-top:5px;'>Identity validated. Greeting user handle: {USER_NAME}</div>"
                time.sleep(0.5)
                root_container.children = [ui]  # Mount main system operations window grid
            else:
                auth_status.value = "<div style='color:#ef4444; font-family:system-ui; font-size:12px; margin-top:5px;'>Verification Failed: Passkey credential mismatch.</div>"
        else:
            # First time configuration profile generation pipeline
            registry[email] = password
            with open(USERS_DB_PATH, "w") as f:
                json.dump(registry, f, indent=2)
            USER_NAME = cleaned_name
            auth_status.value = f"<div style='color:#10b981; font-family:system-ui; font-size:12px; margin-top:5px;'>New Profile registered and verified. Welcoming user handle: {USER_NAME}</div>"
            time.sleep(0.5)
            root_container.children = [ui]
    except Exception as e:
        auth_status.value = f"<div style='color:#ef4444; font-family:system-ui; font-size:12px; margin-top:5px;'>Database track IO exception fault: {str(e)}</div>"

submit_auth_btn.on_click(handle_identity_verification)

# Final Execution sequence view delivery
display(root_container)
