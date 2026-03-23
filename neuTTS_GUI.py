from neutts import NeuTTS
import re
import numpy as np
import soundfile as sf
import sounddevice as sd
import queue
import threading
import os
import time
from typing import List
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import trafilatura
from trafilatura import extract
from PyPDF2 import PdfReader
import requests
import warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import torch

# Global playback state
playback_queue = queue.Queue()  # thread-safe FIFO
playing = threading.Event()     # to control playback loop
playback_stop = threading.Event()

def _playback_worker():
    """Background thread: plays chunks from queue sequentially."""
    while not playback_stop.is_set():
        try:
            # Wait up to 0.5s for next chunk (non-blocking)
            wav = playback_queue.get(timeout=0.3)
            if wav is None:  # sentinel to stop
                break
            print("🔊 Playing chunk (start)...")
            # Play and wait (blocking), or use callback if async needed
            sd.play(wav, samplerate=24000)
            sd.wait()  # blocks until playback finishes
            print("🔊 Done.")
            playback_queue.task_done()
        except queue.Empty:
            continue  # nothing to play yet
        except sd.PortAudioError as e:
            print(f"Playback error: {e}")
            break

# Start playback worker thread
player_thread = threading.Thread(target=_playback_worker, daemon=True)
player_thread.start()

# Add this after the global variable declarations but before the GUI setup
def get_available_samples():
    """Scan samples directory for available .wav/.txt pairs"""
    samples_dir = Path("./samples")
    if not samples_dir.exists():
        return []
    
    samples = set()
    for file in samples_dir.glob("*.wav"):
        # Remove .wav extension to get base name
        base_name = file.stem
        # Check if corresponding .txt file exists
        txt_file = samples_dir / f"{base_name}.txt"
        if txt_file.exists():
            samples.add(base_name)
    
    return sorted(list(samples))


# === Setup ===
backbone_repo_EN_path = os.path.abspath("../neutts-nano")
backbone_repo_ES_path = os.path.abspath("../neutts-nano-spanish")
backbone_repo_FR_path = os.path.abspath("../neutts-nano-french")
backbone_repo_DE_path = os.path.abspath("../neutts-nano-german")

codec_repo_path = os.path.abspath("/../neucodec")

ref_audio_path = "samples/dave.wav"
ref_text_path = "samples/dave.txt"

tts_actual_language = "en"
tts_actual_compute = "cuda"


# Ensure audio and text files exist
if not os.path.exists(ref_audio_path):
    raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
if not os.path.exists(ref_text_path):
    raise FileNotFoundError(f"Reference text not found: {ref_text_path}")

# Setup neuTTS
tts = NeuTTS(
    backbone_repo=backbone_repo_EN_path,
    backbone_device=tts_actual_compute,
    # codec_repo=codec_repo_path,
    codec_repo="neuphonic/neucodec",
    codec_device=tts_actual_compute,
    language="en-us"
   )


def update_tts(params: dict):
    global tts, backbone_repo_path, tts_actual_language, tts_actual_compute, backbone_repo_EN_path
    lang = params.get('lang', tts_actual_language)
    compute_type = params.get('compute_type', tts_actual_compute)  # default value

    if lang == "en":
        backbone_repo_path = os.path.abspath(backbone_repo_EN_path)
        language_setting = "en-us"
    if lang == "es":
        backbone_repo_path = os.path.abspath(backbone_repo_ES_path)
        language_setting = "es-es"
    if lang == "fr":
        backbone_repo_path = os.path.abspath(backbone_repo_FR_path)
        language_setting = "fr-fr"
    if lang == "de":
        backbone_repo_path = os.path.abspath(backbone_repo_DE_path)
        language_setting = "de-de"

    # add more as you need

    tts = NeuTTS(
        backbone_repo=backbone_repo_path,
        backbone_device=compute_type,
        language=language_setting
    )

    tts_actual_language = lang
    tts_actual_compute = compute_type

    return tts

def smart_chunk_text(
    text: str,
    target_len: int = 120,
    max_len: int = 145,
    min_len: int = 15  # NEW: prevent micro-chunks
) -> list[str]:
    """
    Chunk text into speech-friendly segments.
    
    Strategy:
      - Always split on newlines (strong boundary)
      - Prefer sentence-ending punctuation (., ;, ? !) only if chunk ≥ min_len
      - Fall back to comma/semicolon if no sentence break found
      - Then fallback to last space ≤ max_len
      - Hard cut only as last resort (and avoid mid-word if possible)
      - Never exceed max_len; always respect min_len (if possible)
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    pause_marker = ' ... '

    # Sentence-ending punctuation (strong boundary)
    sentence_ends = {'.', '?', '!'}
    # Clause-level punctuation (weaker boundary)
    clause_ends = {',', ';', ':'}
    # All preferred separators
    all_punct = sentence_ends | clause_ends

    # Pre-split on newlines to respect strong paragraph breaks
    # We'll process each line separately, then re-chunk *within* lines
    lines = re.split(r'(\n)', text)  # Keep delimiters

    for i, part in enumerate(lines):
        if not part:
            continue

        # Removes leading whitespace only
        part = part.strip()
        
        # If this part is a newline (odd index in split result), just add empty line break
        if part == '\n':
            continue

        # Process non-newline text
        segment_start = 0
        segment = part

        while segment_start < len(segment):
            # Remaining in this line
            remaining = len(segment) - segment_start
            if remaining <= max_len:
                chunk = segment[segment_start:].rstrip()
                if chunk:
                    cleaned_chunk = re.sub(r' :', ',', chunk)
                    cleaned_chunk = re.sub(r' ;', ',', cleaned_chunk)
                    cleaned_chunk = cleaned_chunk.strip()

                    if i + 1 < len(lines) and lines[i + 1] == '\n':
                        # Next token is a newline → add pause marker after this text
                        chunks.append(cleaned_chunk + pause_marker)
                    else:
                        # End of text, or followed by more text (no newline) → keep as is
                        chunks.append(cleaned_chunk)

                    # chunks.append(cleaned_chunk)
                break

            # Search window: up to max_len ahead
            search_end = min(segment_start + max_len, len(segment))
            candidate = segment[segment_start:search_end]

            best_cut = None

            # Step 1: Try sentence-ending punctuation first (strongest preference)
            for pos in range(min(segment_start + target_len, len(segment)) - 1, segment_start - 1, -1):
                if pos >= len(segment):
                    continue
                ch = candidate[pos - segment_start]
                if ch in sentence_ends:
                    # Check min_len before accepting
                    chunk_len = pos - segment_start + 1  # include punctuation
                    if chunk_len >= min_len or chunk_len >= len(candidate):
                        best_cut = pos + 1  # cut after punctuation
                        break

            # Step 2: If no sentence end, try clause separators (, ;)
            if best_cut is None:
                for pos in range(min(segment_start + target_len, len(segment)) - 1, segment_start - 1, -1):
                    if pos >= len(segment):
                        continue
                    ch = candidate[pos - segment_start]
                    if ch in clause_ends:
                        chunk_len = pos - segment_start + 1
                        if chunk_len >= min_len or chunk_len >= len(candidate):
                            best_cut = pos + 1
                            break

            # Step 3: If still no cut, find last space ≤ max_len (avoid word split)
            if best_cut is None:
                # Look backward for space, but respect min_len
                min_search_pos = max(segment_start + min_len - 1, segment_start)
                for pos in range(search_end - 1, min_search_pos - 1, -1):
                    if pos >= len(segment):
                        continue
                    if segment[pos] == ' ':
                        best_cut = pos  # cut before space
                        break

            # Step 4: Fallback to hard cut at max_len (last resort)
            if best_cut is None:
                best_cut = search_end
                # Try to avoid splitting word: move back to last space if within 30 chars
                fallback_margin = 30
                if best_cut - segment_start > min_len + fallback_margin:
                    for pos in range(best_cut - 1, best_cut - fallback_margin - 1, -1):
                        if pos >= len(segment):
                            continue
                        if segment[pos] == ' ':
                            best_cut = pos
                            break

            # Ensure valid cut
            best_cut = max(segment_start + 1, min(best_cut, len(segment)))

            chunk = segment[segment_start:best_cut].rstrip()
            if chunk:
                cleaned_chunk = re.sub(r' :', ',', chunk)
                cleaned_chunk = re.sub(r' ;', ',', cleaned_chunk)
                cleaned_chunk = cleaned_chunk.strip()
                chunks.append(cleaned_chunk)

            segment_start = best_cut

    i = 0
    while i < len(chunks) - 1:
        if len(chunks[i]) < min_len and chunks[i].strip() and not chunks[i].endswith(('.', '?', '!')):
            # Merge with next chunk
            chunks[i] += ' ' + chunks[i + 1]
            del chunks[i + 1]
        else:
            i += 1

    print(f"✅ chunks: {chunks}")

    return chunks


# Url to text_input

# Suppress only the single InsecureRequestWarning from requests
warnings.simplefilter('ignore', InsecureRequestWarning)

def url_to_text(url: str, element_separator: str = "\n") -> str:
    """
    Extract main content from a URL using trafilatura + custom headers.
    SSL verification is *disabled* for extraction (safe for public content).
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15, verify=False)  # ← SSL bypass
        response.raise_for_status()
        html = response.text
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {e}"
    except Exception as e:
        return f"❌ Unexpected error fetching URL: {e}"

    try:
        raw_text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            output_format="txt"
        )
    except Exception as e:
        return f"❌ Extraction error: {e}"

    if not raw_text:
        return "❌ No readable content found (page may be empty, JS-only, or blocked)."

    # Normalize: join paragraphs with single newline
    paragraphs = [p.strip() for p in raw_text.split("\n") if p.strip()]
    return element_separator.join(paragraphs)

def on_extract_url():
    url = url_entry.get().strip()
    if not url:
        log("⚠️ Please enter a valid URL.")
        return

    log(f"📥 Extracting text from: {url[:60]}{'...' if len(url) > 60 else ''}")

    def _do_extraction():
        try:
            extracted_text = url_to_text(url)
            # Log full result (even errors) for visibility
            log(f"📝 Result preview:\n{extracted_text[:300]}{'...' if len(extracted_text) > 300 else ''}")
            
            # If it's an error message (starts with ❌), show in input area too
            if extracted_text.startswith("❌"):
                root.after(0, lambda: text_input.delete("1.0", tk.END))
                root.after(0, lambda: text_input.insert("1.0", extracted_text))
            else:
                root.after(0, lambda: _set_text_in_input(extracted_text))
        except Exception as e:
            error_msg = f"❌ Unexpected error: {e}\n{traceback.format_exc()}"
            log(error_msg)
            root.after(0, lambda: text_input.delete("1.0", tk.END))
            root.after(0, lambda: text_input.insert("1.0", error_msg))

    threading.Thread(target=_do_extraction, daemon=True).start()

def _set_text_in_input(text: str):
    """Thread-safe way to set text in the main input field."""
    text_input.config(state='normal')
    text_input.delete("1.0", tk.END)
    text_input.insert("1.0", text)
    text_input.config(state='disabled')  # optional: make read-only after extraction
    log("✅ Text extracted and loaded into input field.")
    # Re-enable editing (user may want to tweak extracted text)
    text_input.config(state='normal')


# File to text_input
def file_to_text(file_path: str, element_separator: str = "\n") -> str:
    """
    Extract text from a local file (txt, pdf, etc.).
    """
    try:
        if file_path.lower().endswith('.pdf'):
            # For PDF files
            text = ""
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()

        elif file_path.lower().endswith('.txt'):
            # For plain text files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()

        else:
            # Try to read as HTML or generic text using trafilatura
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                raw_text = extract(content, output_format="txt")
                if not raw_text:
                    return "❌ No readable content found."
                return raw_text.strip()

    except Exception as e:
        return f"❌ Error reading file: {e}"




# === GUI Setup ===
root = tk.Tk()
root.title("NeuTTS GUI")
root.geometry("700x950")
root.configure(bg="#f5f5f5")  # Light background for modern look

# Use ttk for consistent styling (if available)
ttk.Style().theme_use('clam')

# --- Main Frame ---
main_frame = ttk.Frame(root, padding="15")
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Text Input Section ---
input_section = ttk.LabelFrame(main_frame, text="📝 Input Text", padding="10")
input_section.pack(fill=tk.BOTH, pady=(0, 12))

input_label = ttk.Label(input_section, text="Enter text to synthesize:")
input_label.pack(anchor="w", pady=(0, 8))

text_input = scrolledtext.ScrolledText(
    input_section, wrap=tk.WORD, width=90, height=22, font=("Segoe UI", 9)
)
text_input.pack(fill=tk.BOTH, expand=True)


sample_section = ttk.LabelFrame(main_frame, text="🎤 Reference Sample", padding="10")
sample_section.pack(fill=tk.X, pady=(0, 12))

# Top row: Sample + Language (side-by-side)
row_frame = ttk.Frame(sample_section)
row_frame.pack(fill=tk.X, pady=(0, 10))

# Sample dropdown (left)
sample_label = ttk.Label(row_frame, text="Sample:", width=8, anchor="e")
sample_label.pack(side=tk.LEFT, padx=(0, 6))

current_sample = tk.StringVar(value="garage2")
available_samples = get_available_samples()
if available_samples:
    current_sample.set(available_samples[0])

sample_dropdown = ttk.Combobox(
    row_frame,
    textvariable=current_sample,
    values=available_samples,
    state="readonly",
    width=25
)
sample_dropdown.pack(side=tk.LEFT, padx=(0, 10))

# Add a refresh button to rescan samples
def refresh_samples():
    global available_samples, sample_dropdown, current_sample
    available_samples = get_available_samples()
    if available_samples:
        # Update combobox values
        sample_dropdown['values'] = available_samples
        # Ensure current selection is valid; if not, pick first item
        if current_sample.get() not in available_samples:
            current_sample.set(available_samples[0])
    else:
        sample_dropdown['values'] = []
        current_sample.set("")



# Language dropdown (right)
lang_label = ttk.Label(row_frame, text="Language:", width=8, anchor="e")
lang_label.pack(side=tk.LEFT, padx=(0, 6))

current_lang = tk.StringVar(value="en")
lang_dropdown = ttk.Combobox(
    row_frame,
    textvariable=current_lang,
    values=["en", "es", "fr", "de"],
    state="readonly",
    width=8
)
lang_dropdown.pack(side=tk.LEFT, padx=(0, 10))
lang_dropdown.set("en")  # sync with StringVar


# compute type
compute_label = ttk.Label(row_frame, text="Compute:", width=8, anchor="e")
compute_label.pack(side=tk.LEFT, padx=(0, 6))

compute_type = tk.StringVar(value="cuda")
compute_dropdown = ttk.Combobox(
    row_frame,
    textvariable=compute_type,
    values=["cuda", "cpu"],
    state="readonly",
    width=8
)
compute_dropdown.pack(side=tk.LEFT, padx=(0, 10))
compute_dropdown.set("cuda")  # sync with StringVar

# Bottom row: Refresh button (aligned left)
btn_frame = ttk.Frame(sample_section)
btn_frame.pack(fill=tk.X, pady=(0, 6))

refresh_btn = ttk.Button(btn_frame, text="🔄 Refresh", command=refresh_samples)
refresh_btn.pack(side=tk.LEFT)


# auto-reload TTS when language changes
def on_lang_change(*args):
    global tts_actual_language
    lang = current_lang.get()
    if lang != tts_actual_language:
        try:
            # set_language(lang)
            update_tts({"lang": lang})
            tts_actual_language = lang
        except Exception as e:
            print(f"⚠️ Failed to switch language to '{lang}': {e}")

current_lang.trace_add("write", on_lang_change)


def on_compute_change(*args):
    global tts_actual_compute
    compute = compute_type.get()
    if compute != tts_actual_compute:
        try:
            update_tts({"compute_type": compute})
            tts_actual_compute = compute
        except Exception as e:
            print(f"⚠️ Failed to switch language to '{compute}': {e}")

compute_type.trace_add("write", on_compute_change)



# --- URL extractor Section ---
url_section = ttk.LabelFrame(main_frame, text="🔗 Extract from URL", padding="10")
url_section.pack(fill=tk.X, pady=(0, 12))

url_frame = ttk.Frame(url_section)
url_frame.pack(fill=tk.X)

url_label = ttk.Label(url_frame, text="Paste URL:")
url_label.pack(side=tk.LEFT, padx=(0, 6))

url_entry = ttk.Entry(url_frame, width=60)
url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
url_entry.insert(0, "https://example.com/article")

extract_btn = ttk.Button(url_frame, text="📄 Extract", command=lambda: threading.Thread(target=on_extract_url, daemon=True).start())
extract_btn.pack(side=tk.LEFT)




# --- File Reader- txt pdf html - Section ---
def select_and_extract_file():
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[
            ("Text files", "*.txt"),
            ("PDF files", "*.pdf"),
            ("HTML files", "*.html"),
            ("All files", "*.*")
        ]
    )
    if not file_path:
        log("⚠️ No file selected.")
        return

    log(f"📥 Extracting text from file: {os.path.basename(file_path)}")

    def _do_extraction():
        try:
            extracted_text = file_to_text(file_path)
            log(f"📝 Result preview:\n{extracted_text[:300]}{'...' if len(extracted_text) > 300 else ''}")

            if extracted_text.startswith("❌"):
                root.after(0, lambda: text_input.delete("1.0", tk.END))
                root.after(0, lambda: text_input.insert("1.0", extracted_text))
            else:
                root.after(0, lambda: _set_text_in_input(extracted_text))
        except Exception as e:
            error_msg = f"❌ Unexpected error: {e}\n{traceback.format_exc()}"
            log(error_msg)
            root.after(0, lambda: text_input.delete("1.0", tk.END))
            root.after(0, lambda: text_input.insert("1.0", error_msg))

    threading.Thread(target=_do_extraction, daemon=True).start()


file_section = ttk.LabelFrame(main_frame, text="🔗 Extract from File", padding="10")
file_section.pack(fill=tk.X, pady=(0, 12))

file_frame = ttk.Frame(file_section)
file_frame.pack(fill=tk.X)

file_button = tk.Button(file_section, text="📁 Select File", command=select_and_extract_file)
file_button.pack(pady=5)

# --- Output & Controls Section ---
output_section = ttk.LabelFrame(main_frame, text="⚙️ Controls", padding="10")
output_section.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

# Save checkbox
save_frame = ttk.Frame(output_section)
save_frame.pack(anchor="w", pady=(0, 8))

save_file = tk.BooleanVar(value=True)
save_checkbox = ttk.Checkbutton(save_frame, text="💾 Save audio to file", variable=save_file)
save_checkbox.pack()

# Log area
log_label = ttk.Label(output_section, text="Activity Log:")
log_label.pack(anchor="w", pady=(0, 4))

log_area = scrolledtext.ScrolledText(
    output_section, wrap=tk.WORD, width=90, height=6, state='disabled', bg="#e8e8e8",
    font=("Consolas", 9)
)
log_area.pack(fill=tk.BOTH, expand=True)

# --- Action Buttons ---
action_frame = ttk.Frame(main_frame)
action_frame.pack(fill=tk.X, pady=(0, 10))

generate_btn = ttk.Button(
    action_frame, text="🔊 Generate & Play",
    command=lambda: threading.Thread(target=on_generate, daemon=True).start()
)
generate_btn.pack(side=tk.LEFT, padx=(0, 10))

stop_btn = ttk.Button(
    action_frame, text="⏹️ Stop Playback",
    command=lambda: playback_stop.set()
)
stop_btn.pack(side=tk.LEFT)


# --- Footer spacing ---
ttk.Frame(main_frame, height=10).pack()  # Bottom padding


# Update the global reference variables to use the selected sample
def update_reference():
    global ref_text, ref_codes
    """Update reference audio and text paths based on selected sample"""
    sample_name = current_sample.get()
    ref_audio_path = f"samples/{sample_name}.wav"
    ref_text_path = f"samples/{sample_name}.txt"
    
    # Ensure files exist
    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
    if not os.path.exists(ref_text_path):
        raise FileNotFoundError(f"Reference text not found: {ref_text_path}")
    
    # Load and encode reference
    print("Loading reference audio and encoding voice...")
    ref_text = open(ref_text_path, "r", encoding="utf-8").read().strip()
    # ref_codes = tts.encode_reference(ref_audio_path)

    ref_pt_path = Path(ref_text_path).with_suffix(".pt")

    # Only encode if .pt file doesn't exist
    if not ref_pt_path.exists():
        print("Encoding reference voice (not found in cache)...")
        ref_codes = tts.encode_reference(ref_audio_path)
        torch.save(ref_codes, ref_pt_path)
    else:
        print(f"Loading cached voice encoding from {ref_pt_path}...")
        ref_codes = torch.load(ref_pt_path, map_location="cuda")
    

    print("✅ Reference voice encoding ready.")
    
    return ref_text, ref_codes

ref_text = ""
ref_codes = None

# Load and encode reference once
ref_text, ref_codes = update_reference()

# Warm-up
try:
    _ = tts.infer("warm up", ref_codes, ref_text)
    print("✅ Model warmed up.")
except Exception as e:
    print(f"⚠️ Warm-up skipped: {e}")


# === Redirect console prints to GUI log ===
def log(msg: str):
    def append():
        log_area.config(state='normal')
        log_area.insert(tk.END, msg + "\n")
        log_area.see(tk.END)
        log_area.config(state='disabled')
    root.after(0, append)  # thread-safe UI update

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U0001F900-\U0001F9FF"
    "]+", flags=re.UNICODE
)

# === Main generate handler (wraps your existing logic) ===
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

def on_generate():
    user_input = text_input.get("1.0", tk.END).strip()
    if not user_input:
        log("⚠️ No text provided.")
        return

    log(f"\n🎙️ Generating for: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
    
    cleaned = re.sub(r'[#*\]\}\(\)]', '', user_input)
    cleaned = re.sub(r'[\"\‘\“\”\']', '', cleaned)
    cleaned = re.sub(r'[\{\[\_]', ' ', cleaned)
    cleaned = re.sub(r' \- ', ' ', cleaned)
    cleaned = re.sub(r' \— ', ' ', cleaned)
    cleaned = re.sub(r'[\-\—]', ' ', cleaned)
    cleaned = emoji_pattern.sub('', cleaned)

    start_time_glob = time.perf_counter()
    audio_segments = []

    try:
        # Clear stop flag before generation
        playback_stop.clear()

        text_chunks = smart_chunk_text(cleaned, 170, 230, 15)
        log(f"✅ Chunks: {len(text_chunks)}")

        for i, chunk in enumerate(text_chunks, start=1):
            log(f"\n⚙️ Generating chunk {i}...")
            log(f"✅ Text: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

            start_time = time.perf_counter()
            wav = tts.infer(chunk, ref_codes, ref_text)

            audio_segments.append(wav.copy())

            # Queue for playback
            playback_queue.put(wav)

            audio_duration = len(wav) / 24000
            inference_time = time.perf_counter() - start_time
            log(f"   ⏱️ Gen: {inference_time:.3f}s | Audio: {audio_duration:.2f}s | Queued")

        # Save full audio
        if audio_segments:
            wav_full = np.concatenate(audio_segments)
        else:
            wav_full = np.array([], dtype=np.float32)

        inference_time_glob = time.perf_counter() - start_time_glob

        # Save file
        if save_file.get():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_text = cleaned.replace(" ", "_")[:50]
            output_path = output_dir / f"tts_{timestamp}_{safe_text}.wav"
            sf.write(str(output_path), wav_full, 24000)
            log(f"\n✅ Saved: {output_path.name}")
        else:
            log("\nℹ️ File saving disabled.")

        audio_duration_glob = len(wav_full) / 24000
        log(f"⏱️ Total gen time: {inference_time_glob:.3f} s")
        log(f"🔊 Audio duration: {audio_duration_glob:.2f} s")

    except Exception as e:
        log(f"❌ Error: {e}")
        import traceback
        log(traceback.format_exc())


# Optional: graceful shutdown
def on_closing():
    playback_stop.set()
    playback_queue.put(None)  # sentinel for player thread
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Run GUI
log("✅ GUI ready! Paste text and click Generate.")

# Add a callback to update reference when sample changes
def on_sample_change(*args):
    try:
        update_reference()
        log("✅ Reference sample updated")
    except Exception as e:
        log(f"❌ Error updating reference: {e}")

current_sample.trace_add('write', on_sample_change)

root.mainloop()
