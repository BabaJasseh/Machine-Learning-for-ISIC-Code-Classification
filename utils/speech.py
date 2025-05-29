"""
Speech recognition utilities
"""

import streamlit as st
import speech_recognition as sr
import base64

def speech_to_text():
    """Convert speech to text using speech recognition"""
    try:
        r = sr.Recognizer()
        st.info("🎙️ Please speak your business description when ready...")
        
        if "recording" not in st.session_state:
            st.session_state.recording = False
        
        col1, col2 = st.columns([1, 3])
        with col1:
            start_button = st.button("🎙️ Start Recording", key="start_recording")
        with col2:
            rec_status = st.empty()
        
        if st.session_state.recording:
            stop_button = st.button("⏹️ Stop Recording", key="stop_recording", type="primary")
        else:
            stop_button = False
        
        audio_placeholder = st.empty()
        
        if start_button:
            st.session_state.recording = True
            rec_status.warning("🔴 Recording... (speak now)")
            
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=10, phrase_time_limit=20)
                
                st.session_state.audio_data = audio.get_wav_data()
                
                audio_b64 = base64.b64encode(st.session_state.audio_data).decode()
                audio_placeholder.markdown(f"""
                    <audio controls>
                        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.session_state.recording = False
                rec_status.success("✅ Recording complete!")
                
                try:
                    text = r.recognize_google(audio)
                    st.session_state.speech_text = text
                    return text
                except sr.UnknownValueError:
                    st.error("Could not understand audio. Please try again.")
                    return None
                except sr.RequestError:
                    st.error("Could not request results from speech recognition service.")
                    return None
        
        elif stop_button:
            st.session_state.recording = False
            rec_status.success("✅ Recording stopped.")
            return None
        
        if "speech_text" in st.session_state:
            return st.session_state.speech_text
            
        return None
        
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return None