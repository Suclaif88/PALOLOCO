import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Prueba WebRTC Cámara")

class Dummy(VideoTransformerBase):
    def transform(self, frame):
        return frame.to_ndarray(format="bgr24")

webrtc_streamer(
    key="test",
    video_processor_factory=Dummy,
    media_stream_constraints={"video": True, "audio": False},
)
