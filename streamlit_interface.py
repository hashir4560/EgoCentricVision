
import streamlit as st
from datetime import datetime, time
from query_interface import query_object_occurrences
import speech_recognition as sr
from query_agent import fetch_from_query
import json
from driver_code import inference
import os
import cv2
import base64
import tempfile


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)

    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('green2.png')


# Function to save uploaded file locally
def save_uploaded_file(uploaded_file):

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as f:
        f.write(uploaded_file.getbuffer())
        return f.name


# Function to delete uploaded file
def delete_file(file_path):

    if os.path.exists(file_path):
        os.remove(file_path)


# Define object options
objects = [
    'phone',
    'headphones',
    'glasses',
    'wallet',
    'keys',
    'chargers',
    'book',
    'watch',
    'bottle',
    'pen'
]


# Select model
models = [
    'Ego4D v8',
    'Custom v8',
    'Custom v10'
]

model_mapping = {
    'Ego4D v8': 'v1',
    'Custom v8': 'v3',
    'Custom v10': 'v2'
}


st.title("Video Querying Interface")


# Toggle between Simple and Voice input
input_mode = st.radio(
    "Input Mode",
    ('Simple', 'Voice')
)


# Model selection available in both modes
selected_model = st.selectbox(
    'Select the model',
    models,
    index=2
)


selected_object = None
start_datetime = None
end_datetime = None


# Simple input mode
if input_mode == 'Simple':

    selected_object = st.selectbox(
        "Select an object",
        objects
    )

    start_time = st.time_input(
        "Start Time",
        value=time(0, 0)
    )

    start_date = st.date_input(
        'Start Date',
        datetime.now().date()
    )

    end_time = st.time_input(
        "End Time",
        value=time(23, 59)
    )

    end_date = st.date_input(
        'End Date',
        datetime.now().date()
    )


# Voice input mode
else:

    if 'recording' not in st.session_state:
        st.session_state.recording = False

    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None

    if 'voice_query' not in st.session_state:
        st.session_state.voice_query = None

    def start_recording():

        st.session_state.recording = True

        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        try:
            with mic as source:
                st.write("Recording... Please speak into the microphone.")

                recognizer.adjust_for_ambient_noise(source)

                audio = recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=15
                )

            st.session_state.audio_data = audio
            st.session_state.voice_query = None

            st.success("Recording finished.")

        except sr.WaitTimeoutError:
            st.error("No speech detected.")

        except OSError as e:
            st.error(f"Microphone error: {e}")

        finally:
            st.session_state.recording = False

    if not st.session_state.recording:

        if st.button("Start Recording"):
            start_recording()

    if st.session_state.audio_data is not None:

        try:
            if st.session_state.voice_query is None:

                recognizer = sr.Recognizer()

                query_text = recognizer.recognize_google(
                    st.session_state.audio_data
                )

                st.write(f"Recognized Text: {query_text}")

                fetched_query = fetch_from_query(query_text)

                if isinstance(fetched_query, str):
                    query_dict = json.loads(fetched_query)
                else:
                    query_dict = fetched_query

                st.session_state.voice_query = query_dict

            query_dict = st.session_state.voice_query

            selected_object = query_dict['object']
            start_datetime = query_dict['start_datetime']
            end_datetime = query_dict['end_datetime']

            st.write("Selected Object:", selected_object)
            st.write("Start Time:", start_datetime)
            st.write("End Time:", end_datetime)

        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")

        except sr.RequestError as e:
            st.error(f"Speech Recognition error: {e}")

        except (ValueError, KeyError, TypeError) as e:
            st.error(f"Voice query processing error: {e}")


# Toggle for real-time video or upload
mode = st.radio(
    "Mode",
    ('Realtime Video', 'Upload')
)


# File upload input
uploaded_file = None

if mode == 'Upload':

    uploaded_file = st.file_uploader(
        "Upload a video or image",
        type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png']
    )


# Submit button
if st.button("Submit"):

    if input_mode == 'Simple':

        start_datetime = datetime.combine(
            start_date,
            start_time
        ).isoformat()

        end_datetime = datetime.combine(
            end_date,
            end_time
        ).isoformat()

    if (
        selected_object is None
        or start_datetime is None
        or end_datetime is None
    ):
        st.error("Please provide a valid object and date range.")
        st.stop()

    if datetime.fromisoformat(start_datetime) > datetime.fromisoformat(end_datetime):
        st.error("Start date/time must be before end date/time.")
        st.stop()

    st.write(f"Selected Object: {selected_object}")
    st.write(f"Start Time: {start_datetime}")
    st.write(f"End Time: {end_datetime}")

    detection_result = None

    # Uploaded image or video
    if mode == 'Upload':

        if uploaded_file is None:

            st.warning("Please upload a video or image.")
            st.stop()

        st.write(f"Uploaded File: {uploaded_file.name}")

        file_path = save_uploaded_file(uploaded_file)

        try:
            with st.spinner("Running Inference..."):

                detection_result = inference(
                    model_version=model_mapping[selected_model],
                    video_source=file_path
                )

        except Exception as e:
            st.error(f"Inference failed: {e}")

        finally:
            delete_file(file_path)

    # Realtime webcam detection
    elif mode == 'Realtime Video':

        try:
            with st.spinner("Running webcam inference..."):

                detection_result = inference(
                    model_version=model_mapping[selected_model],
                    video_source=0,
                    frame_count=100
                )

        except Exception as e:
            st.error(f"Webcam inference failed: {e}")

    # Display detection results
    if detection_result is not None:

        detections = detection_result['detections']
        annotated_frame = detection_result['frame']

        st.subheader("YOLO Detection Result")

        if annotated_frame is not None:

            rgb_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                rgb_frame,
                caption="YOLO Detection Result",
                width="stretch"
            )

        if detections:

            st.success(
                f"Detected {len(detections)} object occurrence(s)"
            )

            for detection in detections:

                st.write(
                    f"Object: {detection['object']} | "
                    f"Confidence: {detection['confidence']:.2%}"
                )

            matching_current = [
                detection
                for detection in detections
                if detection['object'].lower()
                == selected_object.lower()
            ]

            if matching_current:

                st.success(
                    f"Selected object '{selected_object}' "
                    "detected in current input."
                )

                latest_match = matching_current[-1]

                snapshot_path = os.path.join(
                    "snapshots",
                    f"{latest_match['frame_id']}.jpg"
                )

                if os.path.exists(snapshot_path):

                    st.image(
                        snapshot_path,
                        caption=f"Detected: {selected_object}",
                        width="stretch"
                    )

            else:

                st.info(
                    f"Selected object '{selected_object}' "
                    "was not detected in current input."
                )

        else:

            st.warning(
                "No objects detected in the uploaded file or video."
            )

    # Query historical database
    st.subheader("Database Query Results")

    results = query_object_occurrences(
        object_class=selected_object,
        start_time=start_datetime,
        end_time=end_datetime
    )

    if results:

        st.write("Matching frame_id:", results[-1])

        snapshot_path = os.path.join(
            "snapshots",
            f"{results[-1]}.jpg"
        )

        if os.path.exists(snapshot_path):

            st.image(
                snapshot_path,
                caption=f"Found: {selected_object}",
                width="stretch"
            )

        else:

            st.warning(
                "Matching database record found, "
                "but the snapshot file is missing."
            )

    else:

        st.write("No matching historical object found.")
