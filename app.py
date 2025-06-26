import streamlit as st
import base64

st.title("🎵 Demo Cámara HTML5 + Sonidos")

st.markdown("""
<video id="video" width="400" autoplay playsinline muted></video>
<script>
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      document.getElementById('video').srcObject = stream;
    });
</script>
""", unsafe_allow_html=True)

# Simular sonidos al presionar botones
sound_files = [
    "sounds/#fa.wav",
    "sounds/la.wav",
    "sounds/re.wav",
    "sounds/#do.wav",
    "sounds/#sol.wav"
]

def play_audio(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True
        )

st.markdown("Presiona los botones para simular sonido por dedo:")

for i, s in enumerate(sound_files):
    if st.button(f"Dedos {i+1}"):
        play_audio(s)
