import os, json, datetime, threading, tempfile, time
import numpy as np
import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.signal
import speech_recognition as sr
import easyocr
from PIL import Image
from gtts import gTTS
import pygame
import ollama
import ipywidgets as widgets
from IPython.display import display, HTML as DHTML
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

BASE_DIR = r"C:\Users\Aditya Ranjan\Desktop\MNNIT_Chatbot"
SAVE_DIR = os.path.join(BASE_DIR, "chatbot_data")
CONV_DIR = os.path.join(BASE_DIR, "chatbot_conversations")
DB_DIR   = os.path.join(BASE_DIR, "chatbot_db")
for d in [SAVE_DIR, CONV_DIR, DB_DIR]:
    os.makedirs(d, exist_ok=True)

USER_NAME   = "Adi"
STATE       = {"voice_mode": 0, "is_speaking": False}
stop_speech = threading.Event()
cam_stop    = threading.Event()

MIC_DEVICE  = 27
SAMPLE_RATE = 44100

WORDS = ['hello','hi','hey','college','hostel','mess','campus',
         'professor','consulting','strategy','market','business',
         'case','gym','workout','protein','fitness','health',
         'yes','no','okay','thanks','sorry','help','stop',
         'good','bad','please','welcome','goodbye']

json_path  = os.path.join(SAVE_DIR, "manual_knowledge.json")
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

pygame.mixer.init()
print("✅ CELL 1 DONE")

#cell 2

print("Loading EasyOCR on GPU...")
ocr_reader = easyocr.Reader(['en', 'hi'], gpu=True)
print("✅ EasyOCR ready")

print("Loading vector DB...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb   = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
print("✅ Vector DB ready")

conversation_history = []
conversation_log     = []

def reload_kb():
    global manual_knowledge
    try:
        with open(json_path, "r") as f:
            manual_knowledge = json.load(f)
    except: pass

def detect_domain(q):
    q = q.lower()
    n = sum(1 for k in ['nit','mnnit','allahabad','professor','faculty',
                        'hostel','mess','campus','college','exam',
                        'placement','department','fest','senior',
                        'junior','ta','prayagraj','culrav','avishkar'] if k in q)
    c = sum(1 for k in ['consulting','case','mckinsey','bcg','bain',
                        'strategy','framework','market','business',
                        'mece','porter','revenue','profit','client'] if k in q)
    h = sum(1 for k in ['gym','workout','fitness','protein','diet',
                        'exercise','muscle','health','calories','sleep',
                        'supplement','creatine','bulk','nutrition',
                        'recovery','weight'] if k in q)
    scores = {'nit_allahabad':n, 'consultancy':c, 'health_gym':h}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'general'

def get_sys(domain):
 return {
        'nit_allahabad': (
            "You are an expert assistant for MNNIT Allahabad. "
            "You know everything about the college including hostels "
            "mess departments faculty placements and campus life. "
            "Always use the provided knowledge base facts. Be friendly and specific."),
        'consultancy': (
            "You are an elite management consulting expert from McKinsey BCG Bain. "
            "Excel at case interviews strategy and frameworks. "
            "Use MECE principles and structured thinking."),
        'health_gym': (
            "You are a certified fitness coach and nutritionist. "
            "Give evidence based practical advice on training nutrition and recovery. "
            "Be motivating and science-backed."),
        'general': (
            "You are a helpful expert assistant specialised in "
            "MNNIT Allahabad consulting and fitness topics.")
    }.get(domain, "You are a helpful assistant.")

def chat(question, uname=USER_NAME):
    reload_kb()  # ← always picks up latest KB editor changes

    domain = detect_domain(question)

    # Vector DB context
    try:
        docs = vectordb.similarity_search(question, k=4)
        vector_ctx = "\n".join([d.page_content for d in docs])
except:
        vector_ctx = ""

    # Manual KB context — ALL domains so nothing is missed
    manual_ctx = ""
    for v in manual_knowledge.values():
        manual_ctx += "\n".join(v) + "\n"

    hist = "".join(
        f"{m['role'].upper()}: {m['content']}\n"
        for m in conversation_history[-8:])

    prompt = (
        f"VECTOR DATABASE CONTEXT:\n{vector_ctx}\n\n"
        f"KNOWLEDGE BASE FACTS (always use these):\n{manual_ctx}\n\n"
        f"CONVERSATION HISTORY:\n{hist}\n\n"
        f"USER ({uname}): {question}\n\n"
        f"Answer using the knowledge base. Be specific and helpful. "
        f"Keep under 100 words for voice readability.")

    conversation_history.append({'role':'user','content':question})

    try:
        resp   = ollama.chat(
            model='llama3',
            messages=[
                {'role':'system', 'content':get_sys(domain)},
                {'role':'user',   'content':prompt}
            ])
        answer = resp['message']['content']
    except Exception as e:
        answer = f"Ollama error: {e}. Please run ollama serve in CMD."

    conversation_history.append({'role':'assistant','content':answer})
    conversation_log.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'domain':    domain,
        'question':  question,
        'answer':    answer
    })
    with open(os.path.join(
            CONV_DIR, f'conv_{datetime.date.today()}.json'), 'w') as f:
        json.dump(conversation_log, f, indent=2)

    return answer, domain

print("✅ CELL 2 DONE")

# cell 3

def _play(text, lang, tld):
    stop_speech.clear()
    STATE["is_speaking"] = True
    tmp_files = []
    try:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()][:5]
        if not sentences:
            sentences = [text.strip()]
        for sentence in sentences:
            if stop_speech.is_set(): break
            tts  = gTTS(text=sentence, lang=lang, tld=tld, slow=False)
            tmp  = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            path = tmp.name
            tmp.close()
            tts.save(path)
            tmp_files.append(path)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if stop_speech.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
            time.sleep(0.1)
    except Exception as e:
        print(f"TTS error: {e}")
    finally:
        STATE["is_speaking"] = False
        for f in tmp_files:
            try: os.unlink(f)
            except: pass

def speak(text):
    stop_tts()
    lang = 'hi' if STATE["voice_mode"] == 1 else 'en'
    tld  = 'com' if STATE["voice_mode"] == 1 else 'co.in'
    threading.Thread(target=_play, args=(text,lang,tld), daemon=True).start()

def stop_tts():
    stop_speech.set()
    try: pygame.mixer.music.stop()
    except: pass
    STATE["is_speaking"] = False

print("✅ CELL 3 DONE")

# cell 4

face_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

UI_JUNK = ['recording','press stop','cancel','speak now','chatbot',
           'ready','ollama','send','clear','mic','ocr','lips','stop',
           'hindi','english','voice','taco','0.0s','type question',
           'press enter','appear here','preview','cell','done',
           'mnnit chatbot','knowledge','editor','unlock','password']

def clean_ocr_text(texts):
    result = []
    for t in texts:
        t = t.strip()
        if len(t) < 4: continue
        if any(j in t.lower() for j in UI_JUNK): continue
        result.append(t)
    return ' '.join(result)

def get_mouth_box(frame, pad=15):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w  = frame.shape[:2]
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
    if len(faces) == 0: return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    fx,fy,fw,fh = faces[0]
    ly = fy + int(fh*0.60)
    fl = gray[ly:fy+fh, fx:fx+fw]
    ms = mouth_cascade.detectMultiScale(fl, 1.5, 15, minSize=(25,15))
    if len(ms) == 0:
        return (max(0,fx+fw//4), max(0,fy+int(fh*0.68)),
                min(w,fx+3*fw//4), min(h,fy+fh))
    ms = sorted(ms, key=lambda m: m[2], reverse=True)
    mx2,my2,mw2,mh2 = ms[0]
    return (max(0,fx+mx2-pad), max(0,ly+my2-pad),
            min(w,fx+mx2+mw2+pad), min(h,ly+my2+mh2+pad))

def analyse_lips(frames, boxes):
    if len(frames) < 5: return None
    motions, prev = [], None
    for frame, box in zip(frames, boxes):
        if box is None: continue
        x1,y1,x2,y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        g = cv2.cvtColor(cv2.resize(crop,(64,32)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            motions.append(np.mean(cv2.absdiff(g, prev)))
        prev = g
    if not motions: return None
    avg   = np.mean(motions)
    mx    = np.max(motions)
    dur   = sum(1 for m in motions if m > avg*1.2)
    score = avg*0.5 + mx*0.3 + dur*0.2
    short  = ['hi','hey','yes','no','okay','bad','good','stop','help']
    medium = ['hello','thanks','sorry','please','welcome','goodbye',
              'college','hostel','mess','campus','gym','case','market','health']
    long_w = ['professor','consulting','strategy','business',
              'workout','protein','fitness','placement']
    cands = (short if score < 4 else (medium if score < 10 else long_w))
    valid = [w for w in cands if w in WORDS] or WORDS
    return valid[int(dur) % len(valid)]

def frame_to_bytes(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    pil.save(buf.name, quality=70)
    buf.close()
    with open(buf.name,'rb') as f: data = f.read()
    try: os.unlink(buf.name)
    except: pass
    return data

def open_camera():
    for idx in [0,1,2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 20)
                return cap
            cap.release()
    return None

def stream_cam(img_widget, seconds=6):   # ← 6s for lips
    cap = open_camera()
    if cap is None:
        print("❌ No camera found")
        return None, None, None
    frames, boxes = [], []
    start = time.time()
    while not cam_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        box = get_mouth_box(frame)
        if box:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,"MOUTH",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
        else:
            cv2.putText(frame,"No face — move closer",(10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        rem = max(0, seconds-(time.time()-start))
        cv2.rectangle(frame,(0,0),(640,32),(0,0,0),-1)
        cv2.putText(frame,
            f"Lip Recording {rem:.1f}s | STOP to cancel",
            (8,22), cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),1)
        frames.append(frame.copy())
        boxes.append(box)
        try: img_widget.value = frame_to_bytes(frame)
        except: pass
        time.sleep(0.04)
        if time.time()-start >= seconds: break
    cap.release()
    return frames, boxes, frames[-1] if frames else None

def listen_mic():
    try:
        pygame.mixer.music.stop()
        time.sleep(0.2)
        print("🎤 Recording...")
        recording = sd.rec(
            int(7 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            device=MIC_DEVICE,
            blocking=True
        )
        print("🎤 Processing...")
        resampled = scipy.signal.resample(
            recording, int(len(recording) * 16000 / SAMPLE_RATE)
        ).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav.write(tmp.name, 16000, resampled)
        tmp.close()
        r = sr.Recognizer()
        with sr.AudioFile(tmp.name) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.record(source)
        try: os.unlink(tmp.name)
        except: pass
        lang = "hi-IN" if STATE["voice_mode"] == 1 else "en-IN"
        text = r.recognize_google(audio, language=lang)
        print(f"🎤 Heard: {text}")
        return text.strip()
    except sr.UnknownValueError:
        print("⚠️ Could not understand"); return ""
    except sr.RequestError as e:
        print(f"❌ API error: {e}"); return ""
    except Exception as e:
        print(f"❌ Mic error: {e}"); return ""

print("✅ CELL 4 DONE")

# cell 5

BADGE = {
    'nit_allahabad': ('🎓 MNNIT',      '#3498db', '#1e3a5f'),
    'consultancy':   ('💼 Consulting',  '#e67e22', '#2d1b00'),
    'health_gym':    ('💪 Fitness',     '#2ecc71', '#0d2818'),
    'general':       ('🤖 General',     '#9b59b6', '#1a0a2e'),
}

def domain_badge(domain):
    label, color, bg = BADGE.get(domain, BADGE['general'])
    return (f"<span style='background:{bg};color:{color};"
            f"padding:3px 10px;border-radius:12px;"
            f"font-size:11px;font-family:monospace;"
            f"border:1px solid {color}'>{label}</span>")

def build_chat_html():
    if not conversation_history:
        return ("<div style='color:#444;text-align:center;"
                "padding:60px 20px;font-size:14px;font-style:italic'>"
                "💬 Ask anything about<br>"
                "<span style='color:#3498db'>🎓 MNNIT</span> &nbsp;"
                "<span style='color:#e67e22'>💼 Consulting</span> &nbsp;"
                "<span style='color:#2ecc71'>💪 Fitness</span>"
                "</div>")
    html   = ""
    pairs  = []
    temp_q = None
    for msg in conversation_history:
        if msg['role'] == 'user':
            temp_q = msg['content']
        elif temp_q:
            pairs.append((temp_q, msg['content']))
            temp_q = None
    for q, a in pairs[-10:]:
        html += (
            f"<div style='margin:12px 0;text-align:right'>"
    f"<span style='background:linear-gradient(135deg,#1e3a5f,#1a2744);"
            f"color:#93c5fd;padding:10px 16px;"
            f"border-radius:18px 18px 4px 18px;"
            f"display:inline-block;max-width:82%;"
            f"font-size:13px;line-height:1.6'>👤 {q}</span></div>")
        html += (
            f"<div style='margin:12px 0'>"
            f"<span style='background:linear-gradient(135deg,#111827,#1f2937);"
            f"color:#e2e8f0;padding:10px 16px;"
            f"border-radius:4px 18px 18px 18px;"
            f"display:inline-block;max-width:82%;"
            f"font-size:13px;line-height:1.6;"
            f"border-left:3px solid #00d4ff'>"
            f"🤖 {a[:700]}</span></div>")
    return html

def st(msg, color="#aaa"):
    icon = ("✅" if "Ready" in msg or "Done" in msg or "Speaking" in msg
            else "🎤" if "Record" in msg or "Heard" in msg
            else "📷" if "OCR" in msg or "Camera" in msg
            else "👄" if "Lip" in msg
            else "⏳" if "Think" in msg or "Transcri" in msg
            else "⏹" if "Stop" in msg
            else "⚠️" if "Nothing" in msg or "Could" in msg
            else "❌" if "error" in msg.lower() or "Error" in msg
            else "🗑️" if "Clear" in msg else "ℹ️")
    return (f"<div style='padding:9px 14px;border-radius:8px;"
            f"background:#111827;color:{color};"
            f"font-family:monospace;font-size:13px;"
            f"border-left:3px solid {color};'>{msg}</div>")

# ── Widgets ───────────────────────────────────────────────────────────────
header = widgets.HTML("""
<div style='background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
padding:22px 26px;border-radius:16px;margin-bottom:12px;
border:1px solid #302b63;box-shadow:0 4px 24px rgba(0,212,255,0.08)'>
  <div style='display:flex;align-items:center;gap:12px;margi
    <div style='width:10px;height:10px;background:#2ecc71;
    border-radius:50%;box-shadow:0 0 8px #2ecc71'></div>
    <h2 style='color:#00d4ff;margin:0;font-family:monospace;
    letter-spacing:3px;font-size:20px'>MNNIT CHATBOT v4.0</h2>
  </div>
  <div style='display:flex;gap:10px;flex-wrap:wrap'>
    <span style='background:#1e3a5f;color:#3498db;padding:5px 14px;
    border-radius:20px;font-size:11px;font-family:monospace;
    border:1px solid #3498db'>🎓 MNNIT Allahabad</span>
    <span style='background:#2d1b00;color:#e67e22;padding:5px 14px;
    border-radius:20px;font-size:11px;font-family:monospace;
    border:1px solid #e67e22'>💼 Consulting</span>
    <span style='background:#0d2818;color:#2ecc71;padding:5px 14px;
    border-radius:20px;font-size:11px;font-family:monospace;
    border:1px solid #2ecc71'>💪 Fitness & Gym</span>
    <span style='background:#1a1a2e;color:#9b59b6;padding:5px 14px;
    border-radius:20px;font-size:11px;font-family:monospace;
    border:1px solid #9b59b6'>🤖 General</span>
  </div>
</div>""")

voice_toggle = widgets.ToggleButtons(
    options=['🇮🇳 Indian English','🕉️ Hindi'],
    value='🇮🇳 Indian English',
    description='Voice:',
    style={'button_width':'160px','description_width':'55px'})

chat_box = widgets.HTML(
    value=(
        "<div style='background:linear-gradient(180deg,#0d0d1a,#0a0a14);"
        "min-height:340px;padding:16px;border-radius:12px'>"
        "<div style='color:#444;text-align:center;padding:60px 20px;"
        "font-size:14px;font-style:italic'>"
        "💬 Ask anything about<br>"
        "<span style='color:#3498db'>🎓 MNNIT</span> &nbsp;"
        "<span style='color:#e67e22'>💼 Consulting</span> &nbsp;"
        "<span style='color:#2ecc71'>💪 Fitness</span>"
        "</div></div>"),
    layout=widgets.Layout(
        height='400px', overflow_y='auto',
        border='1px solid #302b63',
        border_radius='14px',
        margin='0 0 10px 0'))

cam_label = widgets.HTML(
    "<div style='color:#00d4ff;font-size:12px;"
    "font-family:monospace;margin:6px 0'>📷 Live Camera</div>",
    layout=widgets.Layout(display='none'))

cam_img = widgets.Image(
    format='jpeg', width=500, height=320,
    layout=widgets.Layout(
        display='none',
        border='2px solid #00d4ff',
        border_radius='10px',
        margin='4px 0 10px 0'))

text_input = widgets.Text(
    placeholder='Type your question and press Enter or Send...',
    layout=widgets.Layout(width='76%', height='40px'))

send_btn  = widgets.Button(description='💬 Send',  button_style='primary',
                           layout=widgets.Layout(width='110px',height='40px'))
mic_btn   = widgets.Button(description='🎤 Mic',   button_style='success',
                           layout=widgets.Layout(width='110px',height='44px'))
ocr_btn   = widgets.Button(description='📷 OCR',   button_style='warning',
                           layout=widgets.Layout(width='110px',height='44px'))
lip_btn   = widgets.Button(description='👄 Lips',  button_style='info',
                           layout=widgets.Layout(width='110px',height='44px'))
stop_btn  = widgets.Button(description='⏹ STOP',   button_style='danger',
                           layout=widgets.Layout(width='110px',height='44px'))
clear_btn = widgets.Button(description='🗑️ Clear',
                           layout=widgets.Layout(width='110px',height='44px'))
kb_btn    = widgets.Button(description='📚 KB Editor',
         button_style='',
                           layout=widgets.Layout(width='130px',height='44px'))

status_bar = widgets.HTML(
    st("✅ CHATBOT READY — Run: ollama serve in CMD","#2ecc71"))
log_out = widgets.Output(
    layout=widgets.Layout(max_height='50px', overflow_y='auto'))

# ── Helpers ───────────────────────────────────────────────────────────────
def refresh_chat():
    chat_box.value = (
        "<div style='background:linear-gradient(180deg,#0d0d1a,#0a0a14);"
        "min-height:340px;padding:16px;border-radius:12px'>"
        f"{build_chat_html()}</div>")

def show_cam(show=True):
    v = 'block' if show else 'none'
    cam_img.layout.display   = v
    cam_label.layout.display = v

def disable_all():
    for b in [send_btn,mic_btn,ocr_btn,lip_btn]: b.disabled = True

def enable_all():
    for b in [send_btn,mic_btn,ocr_btn,lip_btn]: b.disabled = False

# ── Handlers ─────────────────────────────────────────────────────────────
def on_send(b=None):
    q = text_input.value.strip()
    if not q: return
    text_input.value = ""
    disable_all()
    status_bar.value = st("⏳ Thinking...","#f39c12")
    def _go():
        answer, domain = chat(q)
        refresh_chat()
        label,color,_ = BADGE.get(domain, BADGE['general'])
        status_bar.value = st(f"✅ {label} · 🔊 Speaking...","#2ecc71"
 speak(answer)
        while STATE["is_speaking"]: time.sleep(0.1)
        status_bar.value = st("✅ Ready","#2ecc71")
        enable_all()
    threading.Thread(target=_go, daemon=True).start()

def on_mic(b):
    disable_all()
    lang = "Hindi" if STATE["voice_mode"] == 1 else "Indian English"
    status_bar.value = st(f"🎤 Recording 7s in {lang}... SPEAK NOW","#3498db")
    def _go():
        text = listen_mic()
        if text:
            with log_out: print(f"🎤 Heard: {text}")
            text_input.value = text
            answer, domain = chat(text)
            refresh_chat()
            label,color,_ = BADGE.get(domain, BADGE['general'])
            status_bar.value = st(f"✅ {label} · 🔊 Speaking...","#2ecc71")
            speak(answer)
            while STATE["is_speaking"]: time.sleep(0.1)
            status_bar.value = st("✅ Ready","#2ecc71")
        else:
            status_bar.value = st("⚠️ Nothing heard — speak closer to mic","#e67e22")
        enable_all()
    threading.Thread(target=_go, daemon=True).start()

def on_ocr(b):
    disable_all()
    status_bar.value = st("📷 Opening camera — point at text (8s)...","#f39c12")
    def _go():
        cam_stop.clear()
        show_cam(True)
        # Give 8 seconds live preview then snap
        cap = open_camera()
        if cap is None:
            status_bar.value = st("❌ No camera found","#e74c3c")
            show_cam(False); enable_all(); return
        start = time.time()
        frame = None
        while time.time()-start < 8:           # ← 8s OCR window
            if cam_stop.is_set(): break
            ret, f = cap.read()
            if ret:
                frame = f.copy()
                rem = max(0, 8-(time.time()-start))
                cv2.rectangle(f,(0,0),(640,32),(0,0,0),-1)
                cv2.putText(f, f"OCR in {rem:.1f}s | STOP to cancel",
                            (8,22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,(0,200,255),1)
                try: cam_img.value = frame_to_bytes(f)
                except: pass
            time.sleep(0.04)
        cap.release()

        if frame is None or cam_stop.is_set():
            show_cam(False); enable_all()
            status_bar.value = st("⏹ OCR cancelled","#aaa"); return

        status_bar.value = st("🔍 Running OCR on captured frame...","#f39c12")
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = ocr_reader.readtext(rgb)
        raw    = [r[1] for r in result]
        text   = clean_ocr_text(raw)
        with log_out: print(f"📷 Raw OCR: {raw}\n📷 Clean: {text}")

        if text:
            text_input.value = text
            answer, domain = chat(f"Explain or answer about this text: {text}")
            refresh_chat()
            label,color,_ = BADGE.get(domain, BADGE['general'])
            status_bar.value = st(f"✅ {label} · 🔊 Speaking...","#2ecc71")
            speak(answer)
            while STATE["is_speaking"]: time.sleep(0.1)
        else:
 status_bar.value = st("⚠️ No readable text found — try better lighting","#e67e22")

        show_cam(False)
        status_bar.value = st("✅ Ready","#2ecc71")
        enable_all()
    threading.Thread(target=_go, daemon=True).start()

def on_lip(b):
    disable_all()
    cam_stop.clear()
    show_cam(True)
    status_bar.value = st("👄 Recording lips 6s... SPEAK A WORD NOW","#9b59b6")
    def _go():
        frames, boxes, _ = stream_cam(cam_img, seconds=6)  # ← 6s
        show_cam(False)
        if not frames:
            status_bar.value = st("❌ Camera error","#e74c3c")
            enable_all(); return
        word = analyse_lips(frames, boxes)
        if word:
            with log_out: print(f"👄 Lip word: {word}")
            text_input.value = word
            answer, domain = chat(word)
            refresh_chat()
            label,color,_ = BADGE.get(domain, BADGE['general'])
            status_bar.value = st(
                f"✅ {label} · 👄 '{word}' · 🔊 Speaking...","#2ecc71")
            speak(answer)
            while STATE["is_speaking"]: time.sleep(0.1)
        else:
            status_bar.value = st("⚠️ Could not read lips — try again","#e67e22")
        status_bar.value = st("✅ Ready","#2ecc71")
        enable_all()
    threading.Thread(target=_go, daemon=True).start()

def on_stop(b):
    cam_stop.set()
    stop_tts()
    show_cam(False)
    status_bar.value = st("⏹ Stopped. Ready.","#e74c3c")
    enable_all()

def on_clear(b):
    conversation_history.clear()
    conversation_log.clear()
    text_input.value = ""
    refresh_chat()
    status_bar.value = st("🗑️ Cleared. Ready.","#aaa")

def on_kb(b):
    status_bar.value = st(
        "📚 KB Editor → <a href='http://localhost:5050' "
        "target='_blank' style='color:#00d4ff'>http://localhost:5050</a> "
        "(pw: mnnit2024)","#00d4ff")

def on_voice_change(change):
    STATE["voice_mode"] = 1 if 'Hindi' in change['new'] else 0

voice_toggle.observe(on_voice_change, names='value')
mic_btn.on_click(on_mic)
ocr_btn.on_click(on_ocr)
lip_btn.on_click(on_lip)
stop_btn.on_click(on_stop)
send_btn.on_click(on_send)
clear_btn.on_click(on_clear)
kb_btn.on_click(on_kb)
text_input.on_submit(on_send)

display(widgets.VBox([
    header,
    voice_toggle,
    chat_box,
    cam_label,
    cam_img,
    widgets.HBox([text_input, send_btn],
                 layout=widgets.Layout(margin='6px 0')),
    widgets.HBox([mic_btn, ocr_btn, lip_btn, stop_btn, clear_btn, kb_btn],
                 layout=widgets.Layout(gap='6px', flex_wrap='wrap')),
    status_bar,
    log_out
], layout=widgets.Layout(padding='12px', max_width='880px')))

