"""Demo Streamlit del clasificador de emails.

Dos modos:
  - "Hugging Face (local)": carga el modelo base + adaptador LoRA en memoria.
  - "LM Studio (API)": habla con el servidor local de LM Studio (compatible OpenAI).

Ejecutar:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, load_json
from src.infer import normalize_prediction
from src.prompts import build_messages, build_system

st.set_page_config(page_title="Clasificador de Emails", page_icon="📧", layout="centered")

cfg = load_config("config.yaml")


@st.cache_data
def get_labels():
    return load_json(cfg.paths.labels_file)


labels = get_labels()


# --------------------------------------------------------------------------- #
#  Modo Hugging Face local                                                     #
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_hf():
    from peft import PeftModel

    from src.modeling import load_base_model
    from src.train import make_encoder

    encoder = make_encoder(cfg)
    base = load_base_model(cfg, for_training=False)
    model = PeftModel.from_pretrained(base, cfg.paths.final_adapter_dir)
    model.eval()
    return model, encoder


def predict_hf(email: str):
    from src.infer import predict_one

    model, encoder = load_hf()
    return predict_one(model, encoder, email, labels, cfg, return_raw=True)


# --------------------------------------------------------------------------- #
#  Modo LM Studio (OpenAI-compatible)                                          #
# --------------------------------------------------------------------------- #
def predict_lmstudio(email: str, base_url: str, model_name: str):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="lm-studio")
    messages = [
        {"role": "system", "content": build_system(labels)},
        {"role": "user", "content": email},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=cfg.infer.max_new_tokens,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    return normalize_prediction(raw, labels), raw


# --------------------------------------------------------------------------- #
#  UI                                                                          #
# --------------------------------------------------------------------------- #
st.title("📧 Clasificador de Emails (LoRA)")

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Backend de inferencia", ["Hugging Face (local)", "LM Studio (API)"])
    if mode == "LM Studio (API)":
        base_url = st.text_input("Base URL", "http://localhost:1234/v1")
        model_name = st.text_input("Nombre del modelo", "email-classifier")
    st.caption(f"{len(labels)} etiquetas cargadas")
    with st.expander("Ver etiquetas válidas"):
        st.write(labels)

email = st.text_area("Texto del email", height=220, placeholder="Pega aquí el contenido del email...")

if st.button("Clasificar", type="primary"):
    if not email or not email.strip():
        st.warning("Introduce el texto de un email.")
    else:
        with st.spinner("Clasificando..."):
            try:
                if mode == "Hugging Face (local)":
                    pred, raw = predict_hf(email)
                else:
                    pred, raw = predict_lmstudio(email, base_url, model_name)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error de inferencia: {e}")
                st.stop()

        st.success(f"### Predicción: {pred}")
        with st.expander("Salida cruda del modelo (raw) vs normalizada"):
            st.code(f"raw:        {raw!r}\nnormalizado: {pred}")
            if raw and raw.strip() != pred:
                st.caption("La salida cruda se mapeó a la etiqueta válida más cercana.")
