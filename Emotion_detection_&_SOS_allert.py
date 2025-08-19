"""
Krishi-Mitra AI: Proactive Alert Chatbot (text + browser mic audio)
- Detects explicit suicidal intent in text immediately
- Performs emotion/risk analysis on text and (optional) audio transcript
- Triggers email alerts to the emergency contact via SMTP
- Streamlit single-file app
- Audio is captured in the BROWSER using streamlit-audiorec (no server mic needed)
- Optional server-side transcription with Whisper (ENABLE_WHISPER flag)
"""

import os
import io
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List, Union

import streamlit as st
import soundfile as sf
import numpy as np

# Browser audio recorder component (install: pip install streamlit-audiorec)
from st_audiorec import st_audiorec

# ==============================
# Config
# ==============================
ENABLE_WHISPER = True  # Set True to enable server-side Whisper transcription (requires ffmpeg + openai-whisper)

# ==============================
# Farmer profile (prototype)
# ==============================
FARMER_PROFILE: Dict[str, Any] = {
    "farmer_name": "Person X",
    "village": "SNU",
    "state": "Chennai",
    "emergency_contact_name": "Person Y",
    "emergency_email": "sricheran320@gmail.com",  # email address for emergency contact
}

# ==============================
# Risk detection (text-first)
# ==============================
SUICIDE_KEYWORDS = {
    "suicide", "kill myself", "end it all", "end my life",
    "no way out", "can't go on", "give up", "nothing left",
    "better off dead", "self harm", "hurt myself",
    # Hinglish / Hindi transliterations
    "atmahatya", "apni zindagi khatam", "zindagi khatam", "marna", "mar do", "mai marunga", "mai marungi",
}

NEGATIVE_PATTERNS = [
    r"\bno\s+hope\b",
    r"\bnot\s+worth\s+living\b",
    r"\bcan.?t\s+live\b",
    r"\bwant\s+to\s+die\b",
    r"\bnothing\s+to\s+live\b",
]

def contains_suicidal_keywords(text: str) -> bool:
    t = text.lower()
    if any(kw in t for kw in SUICIDE_KEYWORDS):
        return True
    for pat in NEGATIVE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False

def detect_text_emotion_simple(text: str) -> Dict[str, float]:
    """
    Lightweight heuristic emotion scoring (0..1).
    In a full version, plug a real classifier here.
    """
    t = text.lower()
    scores = {
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "neutral": 0.3,
        "joy": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
    }
    if any(w in t for w in ["sad", "depressed", "dukhi", "dukhi hu", "cry", "ro raha", "ro rahi"]):
        scores["sadness"] += 0.5
    if any(w in t for w in ["angry", "gussa", "furious", "rage"]):
        scores["anger"] += 0.5
    if any(w in t for w in ["afraid", "dar", "darta", "darti", "scared", "anxious"]):
        scores["fear"] += 0.5
    if any(w in t for w in ["happy", "khush", "great", "good", "ok", "theek"]):
        scores["joy"] += 0.3
    for k in scores:
        scores[k] = max(0.0, min(1.0, scores[k]))
    return scores

def analyze_text_risk(
    text: str,
    risk_threshold: float = 0.7
) -> Dict[str, Union[str, bool, List[str]]]:
    """
    Assess risk from text using keyword trigger and heuristic emotions.
    Return a standardized result dict.
    """
    result = {
        "risk_level": "low",  # low/medium/high
        "triggered_emotions": [],
        "keywords_detected": False,
        "alert": False,
        "recommended_action": "Monitor and check in weekly"
    }

    kw_hit = contains_suicidal_keywords(text)
    result["keywords_detected"] = kw_hit

    emotions = detect_text_emotion_simple(text)
    risk_emos = []
    for emo in ["sadness", "anger", "fear"]:
        if emotions.get(emo, 0.0) >= risk_threshold:
            risk_emos.append(emo)

    if kw_hit:
        result["risk_level"] = "high"
        result["triggered_emotions"] = risk_emos
    elif any(emotions.get(e, 0.0) >= 0.5 for e in ["sadness", "anger", "fear"]):
        result["risk_level"] = "medium"
        result["triggered_emotions"] = [e for e in ["sadness", "anger", "fear"] if emotions.get(e, 0.0) >= 0.5]

    result["alert"] = (result["risk_level"] in ("medium", "high")) or kw_hit
    action_map = {
        "high": "Immediate intervention required. Contact emergency support and family right now.",
        "medium": "Urgent follow-up needed. Arrange counselor call within 24 hours.",
        "low": "Continue monitoring. Check in weekly and encourage support."
    }
    result["recommended_action"] = action_map[result["risk_level"]]
    if kw_hit and result["risk_level"] != "high":
        result["recommended_action"] = "Keyword alert. Immediate verification required. Contact support now."

    return result

# ==============================
# Email sending (SMTP)
# ==============================
def send_emergency_email(
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    subject: str,
    body: str
) -> Tuple[bool, str]:
    """
    Send an email via SMTP (STARTTLS).
    Returns (ok, message).
    """
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Email sending failed: {e}"

# ==============================
# Optional server-side STT (Whisper)
# ==============================
def transcribe_whisper_from_bytes(wav_bytes: bytes, lang_hint: Optional[str] = None) -> str:
    """
    Transcribe WAV bytes using Whisper (if enabled).
    Requires: pip install openai-whisper and ffmpeg installed on PATH.
    """
    if not ENABLE_WHISPER:
        return ""
    try:
        import whisper
    except Exception:
        st.error("Whisper not installed. Install with: pip install openai-whisper (and ensure ffmpeg is available).")
        return ""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        model = whisper.load_model("base")  # or "small" for better accuracy
        result = model.transcribe(tmp_path, language=lang_hint) if lang_hint else model.transcribe(tmp_path)
        return (result.get("text") or "").strip()
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return ""

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Krishi-Mitra AI", page_icon="🧭", layout="centered")
st.title("Krishi-Mitra AI")

st.sidebar.header("Farmer Profile (Prototype)")
st.sidebar.json(FARMER_PROFILE)

smtp_server = "smtp.gmail.com" 
smtp_port = 587
smtp_user = "sricharan320@gmail.com"
smtp_password = "plpb jiwq jnli rbzl"


st.markdown("This prototype detects high-risk emotional content and can trigger an email alert to the emergency contact.")

# Tabs
tab_text, tab_audio, tab_log = st.tabs(["📝 Text Check", "🎤 Audio Check (Browser Mic)", "📒 Alert Log"])

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

def trigger_alert(reason: str, risk: str, details: Dict[str, Any]) -> None:
    """
    Append to session alert log and send email (if configured).
    """
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "farmer_name": FARMER_PROFILE["farmer_name"],
        "emergency_contact_name": FARMER_PROFILE["emergency_contact_name"],
        "emergency_email": FARMER_PROFILE["emergency_email"],
        "risk_level": risk,
        "reason": reason,
        "details": details
    }
    st.session_state.alert_log.append(entry)

    to_email = FARMER_PROFILE.get("emergency_email", "").strip()
    if not to_email:
        st.warning("Emergency email is not set in the farmer profile.")
        return

    subject = f"[Krishi-Mitra AI] {risk.upper()} Risk Alert for {FARMER_PROFILE['farmer_name']}"
    body_lines = [
        f"Timestamp: {entry['timestamp']}",
        f"Farmer: {entry['farmer_name']} ({FARMER_PROFILE['village']}, {FARMER_PROFILE['state']})",
        f"Risk Level: {risk.upper()}",
        f"Reason: {reason}",
        "",
        "Details:",
        f"- Triggered Emotions: {details.get('triggered_emotions', [])}",
        f"- Keywords Detected: {details.get('keywords_detected', False)}",
        f"- Recommended Action: {details.get('recommended_action', '')}",
    ]
    if "transcript" in details and details["transcript"]:
        body_lines.append("")
        body_lines.append("Transcript:")
        body_lines.append(details["transcript"])
    body = "\n".join(body_lines)

    if smtp_server and smtp_user and smtp_password and smtp_port:
        ok, msg = send_emergency_email(
            smtp_server=str(smtp_server),
            smtp_port=int(smtp_port),
            smtp_user=str(smtp_user),
            smtp_password=str(smtp_password),
            to_email=to_email,
            subject=subject,
            body=body
        )
        if ok:
            st.success(f"Emergency email sent to {to_email}.")
        else:
            st.error(msg)
    else:
        st.warning("SMTP settings are incomplete. Email not sent.")

with tab_text:
    st.subheader("Text-based Emotion & Risk Analysis")
    text_input = st.text_area("Type or paste the farmer's message", height=120, placeholder="I can't go on... everything is hopeless.")
    risk_threshold = st.slider("High-risk emotion threshold", 0.5, 1.0, 0.7, 0.05)

    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            res = analyze_text_risk(text_input.strip(), risk_threshold=risk_threshold)
            st.write("Result:")
            st.json(res)
            if res["alert"]:
                st.error(f"ALERT: {res['risk_level'].upper()} risk detected")
                trigger_alert(
                    reason="Text analysis",
                    risk=res["risk_level"],
                    details={
                        "triggered_emotions": res.get("triggered_emotions", []),
                        "keywords_detected": res.get("keywords_detected", False),
                        "recommended_action": res.get("recommended_action", "")
                    }
                )
            else:
                st.success("No immediate risk detected. Continue supportive monitoring.")

with tab_audio:
    st.subheader("Audio-based Check (Browser Microphone)")
    st.caption("Use the recorder below. It returns WAV bytes directly on this page.")

    wav_audio_bytes = st_audiorec()  # WAV bytes or None

    if st.button("Process Recording"):
        if wav_audio_bytes is None:
            st.error("No recording found. Please record first.")
        else:
            try:
                # Normalize/validate audio
                data, sr = sf.read(io.BytesIO(wav_audio_bytes), dtype="float32", always_2d=False)
                if isinstance(data, np.ndarray) and data.ndim > 1:
                    data = data.mean(axis=1)
                # Re-encode to a clean WAV buffer
                buf = io.BytesIO()
                sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
                wav_clean_bytes = buf.getvalue()
                st.success(f"Audio captured (sr={sr} Hz).")

                transcript = ""
                if ENABLE_WHISPER:
                    with st.spinner("Transcribing with Whisper..."):
                        # You can pass a language hint like "en" or "hi", or None for auto-detect
                        transcript = transcribe_whisper_from_bytes(wav_clean_bytes, None)

                if transcript:
                    st.write("Transcript:")
                    st.write(transcript)
                    res = analyze_text_risk(transcript, risk_threshold=0.7)
                    st.write("Result:")
                    st.json(res)
                    if res["alert"]:
                        st.error(f"ALERT: {res['risk_level'].upper()} risk detected")
                        trigger_alert(
                            reason="Audio transcript (browser mic)",
                            risk=res["risk_level"],
                            details={
                                "transcript": transcript,
                                "triggered_emotions": res.get("triggered_emotions", []),
                                "keywords_detected": res.get("keywords_detected", False),
                                "recommended_action": res.get("recommended_action", "")
                            }
                        )
                    else:
                        st.success("No immediate risk detected from audio.")
                else:
                    st.info("Transcription disabled or not used. Use Text Check tab or enable Whisper.")

            except Exception as e:
                st.error(f"Audio processing failed: {e}")

with tab_log:
    st.subheader("Alert Log (Prototype)")
    if st.session_state.alert_log:
        st.table(st.session_state.alert_log)
    else:
        st.info("No alerts triggered yet.")

st.caption("Note: On remote servers, local microphones are inaccessible to Python. This app records in the browser and sends WAV bytes to the server. For transcription, enable Whisper here or integrate a hosted STT API. Secure SMTP credentials via secrets in production.")



