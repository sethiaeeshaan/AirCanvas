import math
import os
import traceback

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

FRAME_W, FRAME_H = 1280, 720
HEADER_H = 100
NUM_BANDS = 7
BAND_W = FRAME_W // NUM_BANDS

BAND_COLORS = [
    (20, 150, 0),
    (0, 50, 255),
    (210, 50, 0),
    (7, 210, 252),
    (102, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
]
ERASER_INDEX = NUM_BANDS - 1
DEFAULT_BRUSH = 20
ERASER_BRUSH = 80


class HandDetector:
    def __init__(self, mode=False, max_hands=1, complexity=1,
                 detection_confidence=0.7, tracking_confidence=0.7):
        self.results = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            mode, max_hands, complexity,
            detection_confidence, tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list = []

    def find_hands(self, img, draw=True):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if draw and self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_num=0):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            h, w, _ = img.shape
            for lm_id, lm in enumerate(hand.landmark):
                self.lm_list.append([lm_id, int(lm.x * w), int(lm.y * h)])
        return self.lm_list

    def fingers_up(self):
        fingers = []
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1] - 2:
            fingers.append(1)
        else:
            fingers.append(0)
        for n in range(1, 5):
            if self.lm_list[self.tip_ids[n]][2] < self.lm_list[self.tip_ids[n] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


@st.cache_resource
def load_headers():
    folder = "Header"
    overlays = []
    if not os.path.isdir(folder):
        return overlays
    for name in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            continue
        if img.shape[1] != FRAME_W or img.shape[0] != HEADER_H:
            img = cv2.resize(img, (FRAME_W, HEADER_H))
        overlays.append(img)
    return overlays


def band_index(x):
    return min(max(x, 0) // BAND_W, NUM_BANDS - 1)


class AirCanvasProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector()
        self.overlay_list = load_headers()
        self.header = self.overlay_list[0] if self.overlay_list else None
        self.def_color = BAND_COLORS[0]
        self.brush_thickness = DEFAULT_BRUSH
        self.xp, self.yp = 0, 0
        self.draw_canvas = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        self.clear_requested = False
        self.last_error = None
        self.frame_count = 0

    def latest_canvas_rgb(self):
        return cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2RGB)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            return self._process(frame)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return frame

    def _process(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        img = cv2.flip(img, 1)
        img = cv2.GaussianBlur(img, (17, 17), cv2.BORDER_DEFAULT)

        if self.clear_requested:
            self.draw_canvas[:] = 0
            self.clear_requested = False

        img = self.detector.find_hands(img)
        lm_list = self.detector.find_position(img)

        if lm_list:
            x1, y1 = lm_list[8][1:]
            x3, y3 = lm_list[4][1:]
            x4, y4 = (x1 + x3) // 2, (y1 + y3) // 2
            fingers = self.detector.fingers_up()

            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                if y1 < HEADER_H:
                    idx = band_index(x1)
                    if self.overlay_list and idx < len(self.overlay_list):
                        self.header = self.overlay_list[idx]
                    self.def_color = BAND_COLORS[idx]
                    self.brush_thickness = ERASER_BRUSH if idx == ERASER_INDEX else DEFAULT_BRUSH

            if fingers[1] and fingers[2] and fingers[3]:
                if y1 < HEADER_H and band_index(x1) == ERASER_INDEX:
                    self.draw_canvas[:] = 0

            if fingers[1] and not fingers[2] and not fingers[0]:
                cv2.circle(img, (x1, y1), max(self.brush_thickness // 2, 1), self.def_color, cv2.FILLED)
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1
                cv2.line(img, (self.xp, self.yp), (x1, y1), self.def_color, self.brush_thickness)
                cv2.line(self.draw_canvas, (self.xp, self.yp), (x1, y1), self.def_color, self.brush_thickness)
                self.xp, self.yp = x1, y1

            if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                cv2.circle(img, (x1, y1), 15, (0, 0, 40), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 40), cv2.FILLED)
                cv2.line(img, (x1, y1), (x3, y3), (0, 0, 75), 10)
                length = math.hypot(x3 - x1, y3 - y1)
                self.brush_thickness = max(1, int(length / 5))
                cv2.circle(img, (x4, y4), max(int(self.brush_thickness // 1.25), 1),
                           self.def_color, cv2.FILLED)

        if self.header is not None:
            img[0:HEADER_H, 0:FRAME_W] = self.header
        gray = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, inv)
        img = cv2.bitwise_or(img, self.draw_canvas)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def get_ice_servers():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        urls = st.secrets["turn_urls"]
        username = st.secrets["turn_username"]
        credential = st.secrets["turn_credential"]
        servers.append({
            "urls": urls if isinstance(urls, list) else [urls],
            "username": username,
            "credential": credential,
        })
        return servers
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    servers.append({
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turn:openrelay.metered.ca:443?transport=tcp",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    })
    return servers


st.set_page_config(page_title="Air Canvas", layout="wide")
st.title("Air Canvas")
st.caption("Draw in the air with your index finger. Hand-tracking happens server-side via MediaPipe.")

rtc_config = RTCConfiguration({"iceServers": get_ice_servers(), "iceTransportPolicy": "all"})

video_col, side_col = st.columns([3, 1])

with video_col:
    ctx = webrtc_streamer(
        key="air-canvas",
        video_processor_factory=AirCanvasProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with side_col:
    st.subheader("Controls")

    if ctx.state.playing:
        st.success("Streaming")
    elif ctx.state.signalling:
        st.info("Connecting…")
    else:
        st.warning("Click **START** to begin")

    if ctx.video_processor:
        st.metric("Frames processed", ctx.video_processor.frame_count)

        if st.button("Clear canvas", use_container_width=True):
            ctx.video_processor.clear_requested = True

        if st.button("Prepare capture", use_container_width=True):
            rgb = ctx.video_processor.latest_canvas_rgb()
            ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if ok:
                st.session_state["capture_bytes"] = buf.tobytes()

        if "capture_bytes" in st.session_state:
            st.download_button(
                "Download capture.png",
                data=st.session_state["capture_bytes"],
                file_name="capture.png",
                mime="image/png",
                use_container_width=True,
            )

        if ctx.video_processor.last_error:
            with st.expander("Frame processing error", expanded=True):
                st.code(ctx.video_processor.last_error)

with st.expander("Gestures", expanded=False):
    st.markdown(
        "- **Index + middle finger up** over the top bar — select a color or eraser.\n"
        "- **Index finger only** — draw.\n"
        "- **Thumb + index** — adjust brush thickness (distance between them).\n"
        "- **Index + middle + ring** in the eraser slot — clear the canvas."
    )

with st.expander("Connection notes", expanded=False):
    st.markdown(
        "WebRTC needs a direct UDP path between your browser and the server. "
        "If the video never starts on a deployed instance, your network is most "
        "likely blocking peer connections. The app falls back to public TURN "
        "relay servers, but they're rate-limited.\n\n"
        "**For a stable deployment, add your own TURN credentials** "
        "(Twilio, Metered.ca, or self-hosted coturn) to Streamlit Cloud "
        "**Secrets** as `turn_urls`, `turn_username`, `turn_credential`. "
        "No code change required — they'll be picked up automatically."
    )
