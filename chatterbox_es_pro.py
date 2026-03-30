"""
Chatterbox Español Pro — Multilingüe con clonación de voz y perfiles guardados.
Optimizado para español con soporte completo de 23 idiomas.
"""

import os
import json
import time
import shutil
import random
import numpy as np
import torch
import torchaudio as ta
import torchaudio.functional as F_audio
import librosa
import gradio as gr
import urllib.request
import tempfile
from pathlib import Path
from datetime import datetime
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# --- Action Tags & Effects ---
EVENT_TAGS = {
    "[sigh]": "😮‍💨 Suspiro",
    "[gasp]": "😱 Susto",
    "[groan]": "😫 Quejido",
    "[cough]": "🤒 Tos",
    "[clear throat]": "🗣️ Carraspeo",
    "[chuckle]": "😏 Risita",
    "[laugh]": "😂 Risa",
    "[sniff]": "🤧 Olfateo",
    "[shush]": "🤫 Shh"
}

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val; 

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    let prefix = " ";
    let suffix = " ";

    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";

    if (end < current_text.length && current_text[end] === ' ') suffix = "";

    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

def apply_radio_effect(wav, sr):
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
    # Bandpass (300-3000Hz) to simulate walkie-talkie
    wav = F_audio.highpass_biquad(wav, sr, 300.0)
    wav = F_audio.lowpass_biquad(wav, sr, 3000.0)
    # Distortion
    wav = wav * 2.5
    wav = torch.clamp(wav, min=-1.0, max=1.0)
    return wav.squeeze(0).numpy()

def apply_pitch_shift(wav_np, sr, n_steps):
    return librosa.effects.pitch_shift(wav_np, sr=sr, n_steps=n_steps)

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOICES_DIR = Path("saved_voices")
OUTPUTS_DIR = Path("generated_audio")
VOICES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

print(f"🚀 Dispositivo: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

# --- Model ---
MODEL = None

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("⏳ Cargando modelo multilingüe (puede tardar ~30s la primera vez)...")
        start = time.time()
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        elapsed = time.time() - start
        print(f"✅ Modelo cargado en {elapsed:.1f}s")
    return MODEL

# --- Voice Profile Management ---
def get_voice_profiles():
    """Returns list of saved voice profile names."""
    profiles = []
    if VOICES_DIR.exists():
        for p in sorted(VOICES_DIR.iterdir()):
            if p.is_dir():
                meta_file = p / "meta.json"
                if meta_file.exists():
                    profiles.append(p.name)
    return profiles

def save_voice_profile(name: str, audio_path: str, description: str = ""):
    """Save a voice reference clip as a named profile."""
    if not name or not name.strip():
        return "❌ El nombre del perfil no puede estar vacío.", gr.update()
    if not audio_path:
        return "❌ Debes subir un audio de referencia primero.", gr.update()
    
    name = name.strip().replace(" ", "_")
    profile_dir = VOICES_DIR / name
    profile_dir.mkdir(exist_ok=True)
    
    # Copy the audio file
    ext = Path(audio_path).suffix or ".wav"
    dest = profile_dir / f"reference{ext}"
    shutil.copy2(audio_path, dest)
    
    # Save metadata
    meta = {
        "name": name,
        "description": description,
        "created": datetime.now().isoformat(),
        "reference_file": str(dest),
        "original_file": os.path.basename(audio_path),
    }
    with open(profile_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    profiles = get_voice_profiles()
    return f"✅ Perfil de voz '{name}' guardado correctamente.", gr.update(choices=profiles, value=name)

def delete_voice_profile(name: str):
    """Delete a saved voice profile."""
    if not name:
        return "❌ Selecciona un perfil para eliminar.", gr.update()
    profile_dir = VOICES_DIR / name
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        profiles = get_voice_profiles()
        return f"🗑️ Perfil '{name}' eliminado.", gr.update(choices=profiles, value=None)
    return f"❌ Perfil '{name}' no encontrado.", gr.update()

def get_profile_audio_path(profile_name: str) -> str | None:
    """Get the reference audio path for a saved profile."""
    if not profile_name:
        return None
    profile_dir = VOICES_DIR / profile_name
    meta_file = profile_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("reference_file")
    return None

def load_profile_info(profile_name: str):
    """Load profile info for display."""
    if not profile_name:
        return "", None
    profile_dir = VOICES_DIR / profile_name
    meta_file = profile_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        info = f"📋 **{meta['name']}**\n📅 Creado: {meta['created'][:10]}\n📝 {meta.get('description', 'Sin descripción')}"
        audio_path = meta.get("reference_file")
        return info, audio_path
    return "Perfil no encontrado", None

# --- TTS Generation ---
def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_speech(
    text: str,
    language_id: str,
    voice_source: str,
    uploaded_audio: str,
    saved_profile: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    seed_num: int,
    auto_save: bool,
    repetition_penalty: float,
    top_p: float,
    min_p: float,
    radio_effect: bool,
    pitch_steps: float,
):
    """Generate speech and return audio + stats."""
    if not text.strip():
        raise gr.Error("❌ Escribe algo de texto primero.")
    
    model = get_or_load_model()
    
    # Determine audio prompt
    audio_prompt = None
    voice_label = "voz por defecto"
    
    if voice_source == "📁 Perfil guardado" and saved_profile:
        audio_prompt = get_profile_audio_path(saved_profile)
        voice_label = f"perfil: {saved_profile}"
    elif voice_source == "🎤 Audio subido/grabado" and uploaded_audio:
        audio_prompt = uploaded_audio
        voice_label = "audio subido"
    elif voice_source == "🌐 Voz por defecto del idioma":
        # Use language default from the demo samples
        defaults = {
            "es": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
            "en": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
            "fr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
            "de": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
            "it": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
            "pt": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        }
        audio_prompt = defaults.get(language_id)
        voice_label = f"voz por defecto ({language_id})"
    
    if seed_num != 0:
        set_seed(int(seed_num))
    
    # Generate
    text_truncated = text[:300]
    start_time = time.time()
    
    generate_kwargs = {
        "exaggeration": exaggeration,
        "temperature": temperature,
        "cfg_weight": cfg_weight,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "min_p": min_p,
    }
    if audio_prompt:
        if audio_prompt.startswith("http"):
            # Download to a temporary file, as librosa/torchaudio might fail with URLs on Windows
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".flac").name
            urllib.request.urlretrieve(audio_prompt, temp_file)
            generate_kwargs["audio_prompt_path"] = temp_file
        else:
            generate_kwargs["audio_prompt_path"] = audio_prompt
    
    wav = model.generate(text_truncated, language_id=language_id, **generate_kwargs)
    gen_time = time.time() - start_time
    
    # Post-processing
    wav_np = wav.squeeze(0).cpu().numpy()
    
    if pitch_steps != 0.0:
        wav_np = apply_pitch_shift(wav_np, model.sr, pitch_steps)
        
    if radio_effect:
        wav_np = apply_radio_effect(wav_np, model.sr)
    
    # Calculate stats
    audio_duration = wav_np.shape[0] / model.sr
    rtf = gen_time / audio_duration if audio_duration > 0 else 0
    chars_per_sec = len(text_truncated) / gen_time if gen_time > 0 else 0
    
    stats = (
        f"⏱️ **Tiempo de generación:** {gen_time:.1f}s\n"
        f"🔊 **Duración del audio:** {audio_duration:.1f}s\n"
        f"⚡ **RTF (Real-Time Factor):** {rtf:.2f}x\n"
        f"📝 **Caracteres procesados:** {len(text_truncated)} ({chars_per_sec:.0f} chars/s)\n"
        f"🎙️ **Voz usada:** {voice_label}\n"
        f"🌍 **Idioma:** {SUPPORTED_LANGUAGES.get(language_id, language_id)}\n"
        f"🖥️ **Dispositivo:** {DEVICE}"
    )
    
    # Auto-save output
    if auto_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = OUTPUTS_DIR / f"{language_id}_{timestamp}.wav"
        ta.save(str(out_file), torch.from_numpy(wav_np).unsqueeze(0), model.sr)
        stats += f"\n💾 **Guardado en:** `{out_file}`"
    
    return (model.sr, wav_np), stats


# --- Language display ---
LANG_CHOICES = [(f"{name} ({code})", code) for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])]

# --- CSS ---
CSS = """
.main-header { text-align: center; margin-bottom: 10px; }
.profile-card { padding: 10px; border-radius: 8px; background: #f0f4ff; border: 1px solid #c7d2fe; }
.stats-box { padding: 12px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; }
.tag-container { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 5px !important; margin-bottom: 10px !important; border: none !important; background: transparent !important; }
.tag-btn { min-width: fit-content !important; width: auto !important; height: 32px !important; font-size: 13px !important; background: #eef2ff !important; border: 1px solid #c7d2fe !important; color: #3730a3 !important; border-radius: 6px !important; padding: 0 10px !important; margin: 0 !important; box-shadow: none !important; }
.tag-btn:hover { background: #c7d2fe !important; transform: translateY(-1px); }
"""

# --- Build UI ---
with gr.Blocks(title="Chatterbox Español Pro", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        """
        # 🎙️ Chatterbox Español Pro
        ### Text-to-Speech multilingüe con clonación de voz · 23 idiomas · Perfiles guardados
        """,
        elem_classes=["main-header"]
    )
    
    with gr.Tabs():
        # ============ TAB 1: GENERAR ============
        with gr.Tab("⚡ Generar voz") as tab_generate:
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        value="[sigh] Cerebros... necesito cerebros... [groan]",
                        label="📝 Texto a sintetizar (máx. 300 caracteres)",
                        max_lines=5,
                        placeholder="Escribe aquí el texto que quieres convertir en voz...",
                        elem_id="main_textbox"
                    )
                    
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag_code, tag_label in EVENT_TAGS.items():
                            btn = gr.Button(f"{tag_label} {tag_code}", elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[gr.State(tag_code), text_input], outputs=text_input, js=INSERT_TAG_JS)
                    
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=LANG_CHOICES,
                            value="es",
                            label="🌍 Idioma",
                            scale=1
                        )
                        voice_source = gr.Radio(
                            choices=["🌐 Voz por defecto del idioma", "🎤 Audio subido/grabado", "📁 Perfil guardado"],
                            value="🌐 Voz por defecto del idioma",
                            label="🎙️ Fuente de voz",
                            scale=2
                        )
                    
                    # Conditional voice inputs
                    with gr.Group(visible=False) as upload_group:
                        uploaded_audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="🎤 Sube o graba un audio de referencia (~10 segundos)",
                        )
                    
                    with gr.Group(visible=False) as profile_group:
                        profile_select_gen = gr.Dropdown(
                            choices=get_voice_profiles(),
                            label="📁 Seleccionar perfil de voz guardado",
                            interactive=True,
                        )
                    
                    with gr.Row():
                        exaggeration = gr.Slider(0.25, 4.0, step=0.05, value=0.5, label="🎭 Expresividad (0.5 = neutro)")
                        cfg_weight = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="🎯 CFG/Ritmo (0.5 = normal)")
                    
                    with gr.Accordion("🎬 Efectos de Supervivencia (Post-Procesado)", open=True):
                        with gr.Row():
                            radio_effect = gr.Checkbox(value=False, label="📻 Efecto Walkie-Talkie / Radio")
                            pitch_steps = gr.Slider(-6.0, 6.0, step=0.5, value=0.0, label="⏬ Tono de Voz (Graves/Agudos)")
                            
                    with gr.Accordion("⚙️ Opciones avanzadas", open=False):
                        temperature = gr.Slider(0.05, 5.0, step=0.05, value=0.8, label="🌡️ Temperatura")
                        repetition_penalty = gr.Slider(1.0, 10.0, step=0.1, value=2.0, label="🔄 Penalización por repetición (evita loops / fuerza exhalaciones)")
                        with gr.Row():
                            top_p = gr.Slider(0.1, 1.0, step=0.05, value=1.0, label="🎲 Top P (Restringe tokens posibles)")
                            min_p = gr.Slider(0.0, 1.0, step=0.01, value=0.05, label="🎲 Min P (Corte mínimo de probabilidad)")
                        seed_num = gr.Number(value=0, label="🎲 Semilla (0 = aleatorio)")
                        auto_save = gr.Checkbox(value=True, label="💾 Guardar audio automáticamente")
                    
                    generate_btn = gr.Button("🚀 Generar voz", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    audio_output = gr.Audio(label="🔊 Audio generado", type="numpy")
                    gen_stats = gr.Markdown("*Pulsa 'Generar voz' para empezar...*", elem_classes=["stats-box"])
            
            # Voice source visibility logic
            def toggle_voice_source(source):
                return (
                    gr.update(visible=(source == "🎤 Audio subido/grabado")),
                    gr.update(visible=(source == "📁 Perfil guardado")),
                )
            
            voice_source.change(
                fn=toggle_voice_source,
                inputs=[voice_source],
                outputs=[upload_group, profile_group],
            )
            
            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, language, voice_source, uploaded_audio, profile_select_gen,
                        exaggeration, cfg_weight, temperature, seed_num, auto_save,
                        repetition_penalty, top_p, min_p, radio_effect, pitch_steps],
                outputs=[audio_output, gen_stats],
            )
        
        # ============ TAB 2: PERFILES DE VOZ ============
        with gr.Tab("👤 Perfiles de voz") as tab_profiles:
            gr.Markdown(
                """
                ### Guardar y gestionar perfiles de voz
                Sube un audio de referencia (~10 segundos) y guárdalo como perfil para reutilizarlo.
                Las voces clonadas se guardan en la carpeta `saved_voices/`.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ➕ Crear nuevo perfil")
                    profile_name = gr.Textbox(
                        label="Nombre del perfil",
                        placeholder="ej: mi_voz, narrador_pro, cliente_1..."
                    )
                    profile_desc = gr.Textbox(
                        label="Descripción (opcional)",
                        placeholder="ej: Voz masculina grave, tono profesional"
                    )
                    profile_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="🎤 Audio de referencia (~10 segundos)",
                    )
                    save_btn = gr.Button("💾 Guardar perfil", variant="primary")
                    save_status = gr.Markdown("")
                
                with gr.Column():
                    gr.Markdown("#### 📋 Perfiles guardados")
                    profile_list = gr.Dropdown(
                        choices=get_voice_profiles(),
                        label="Seleccionar perfil",
                        interactive=True,
                    )
                    profile_info = gr.Markdown("", elem_classes=["profile-card"])
                    profile_preview = gr.Audio(label="🔊 Audio de referencia del perfil", interactive=False)
                    delete_btn = gr.Button("🗑️ Eliminar perfil", variant="stop")
                    delete_status = gr.Markdown("")
            
            # Refresh profiles function to easily attach to events
            def refresh_profiles():
                profiles = get_voice_profiles()
                return gr.update(choices=profiles), gr.update(choices=profiles)

            # Save profile
            save_btn.click(
                fn=save_voice_profile,
                inputs=[profile_name, profile_audio, profile_desc],
                outputs=[save_status, profile_list],
            ).then(
                fn=refresh_profiles,
                outputs=[profile_list, profile_select_gen],
            )
            
            # Load profile info
            profile_list.change(
                fn=load_profile_info,
                inputs=[profile_list],
                outputs=[profile_info, profile_preview],
            )
            
            # Delete profile
            delete_btn.click(
                fn=delete_voice_profile,
                inputs=[profile_list],
                outputs=[delete_status, profile_list],
            ).then(
                fn=refresh_profiles,
                outputs=[profile_list, profile_select_gen],
            )

    # Refresh profiles on app load
    demo.load(
        fn=refresh_profiles,
        outputs=[profile_list, profile_select_gen],
    )
    
    tab_generate.select(fn=refresh_profiles, outputs=[profile_list, profile_select_gen])
    tab_profiles.select(fn=refresh_profiles, outputs=[profile_list, profile_select_gen])
        
    # ============ TAB 3: INFO ============
    with gr.Tab("ℹ️ Info y tips"):
            device_info = f"**{DEVICE.upper()}**"
            if DEVICE == "cuda":
                device_info += f" — {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)"
            
            gr.Markdown(f"""
### 🎯 Tips generales

| Parámetro | Recomendación |
|---|---|
| Expresividad | `0.5` para lectura normal, `0.7+` para más drama |
| CFG/Ritmo | `0.5` normal, `0.3` si el hablante es rápido, `0` para cross-language |
| Temperatura | `0.8` por defecto, subir para más variedad |

### 🧟‍♂️ Sobreviviendo a FiveM (Guía de Rol)

El **Chatterbox Español Pro** incluye herramientas exclusivas para servidores de Roleplay postapocalíptico:

1. **Voces de Superviviente Estresado:**
   - Usa los botones de **Acción** (`[gasp]`, `[sigh]`, `[groan]`) al principio o entre frases. Ej: `[gasp] ¡Corre, vienen por nosotros! [groan]`
   - Sube la **Expresividad** a `2.5` - `3.5`.
   - Baja el **Tono de Voz (Pitch)** a `-2.0` para voces graves masculinas rasgadas, o súbelo a `1.5` para voces femeninas temblorosas.
2. **Comunicaciones por Radio/Walkie-Talkie:**
   - Activa la casilla **📻 Efecto Walkie-Talkie** en Opciones Avanzadas.
   - Sube la **Penalización por repetición** a `3.0` o `4.0` para que la voz suene entrecortada y forzada.
   - El filtro recortará los graves y agudos y añadirá saturación para simular estática.

### 🎙️ Clonación de voz — Cómo funciona

1. **Graba o sube un audio** de ~10 segundos de la voz que quieres clonar
2. **El audio debe ser limpio** — sin música de fondo ni ruido
3. **Guárdalo como perfil** en la pestaña "Perfiles de voz" para reutilizarlo
4. El modelo captura el timbre, tono y estilo del hablante de referencia

> 💡 **Importante:** Para español, usa una referencia en español. Si usas otra lengua, el resultado puede tener acento extranjero. Para minimizarlo, pon CFG a 0.

### ⏱️ Tiempos estimados de generación (RTX 3050, 4GB VRAM)

| Longitud del texto | Tiempo estimado | Audio generado |
|---|---|---|
| ~50 caracteres (frase corta) | 3-8s | ~2-4s de audio |
| ~150 caracteres (párrafo corto) | 8-15s | ~5-10s de audio |
| ~300 caracteres (máximo) | 15-30s | ~10-20s de audio |

> ⚠️ La primera generación tarda más porque descarga los pesos del modelo (~2GB).

### 💾 ¿Dónde se guarda todo?

| Qué | Dónde |
|---|---|
| Perfiles de voz | `saved_voices/` (audio + metadatos JSON) |
| Audio generado | `generated_audio/` (WAV, auto-nombrado por fecha) |
| Modelos descargados | Cache de HuggingFace (`~/.cache/huggingface/`) |

### 🖥️ Tu sistema
- **Dispositivo:** {device_info}
- **PyTorch:** {torch.__version__}
- **Python:** {'.'.join(map(str, __import__('sys').version_info[:3]))}
            """)

# --- Launch ---
if __name__ == "__main__":
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7862,
        inbrowser=True
    )
