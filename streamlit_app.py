import os
import math
import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

FRAME_W, FRAME_H = 1280, 720
HEADER_H = 100


class HandDetector:
    def __init__(self, mode=False, maximum_hands=1, complexity=1,
                 detection_confidence=0.85, tracking_confidence=0.85):
        self.results = None
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            mode, maximum_hands, complexity,
            detection_confidence, tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, canvas, draw=True):
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(canvas, handLms, self.mpHands.HAND_CONNECTIONS)
        return canvas

    def findPosition(self, canvas, hand_num=0):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                h, w, _ = canvas.shape
                self.lmList.append([id, int(lm.x * w), int(lm.y * h)])
        return self.lmList

    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1] - 2:
            fingers.append(1)
        else:
            fingers.append(0)
        for n in range(1, 5):
            if self.lmList[self.tipIds[n]][2] < self.lmList[self.tipIds[n] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


@st.cache_resource
def load_headers():
    file_path = "Header"
    overlays = []
    for name in sorted(os.listdir(file_path)):
        img = cv2.imread(os.path.join(file_path, name))
        if img is None:
            continue
        if img.shape[1] != FRAME_W or img.shape[0] != HEADER_H:
            img = cv2.resize(img, (FRAME_W, HEADER_H))
        overlays.append(img)
    return overlays


class AirCanvasProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector()
        self.overlayList = load_headers()
        self.header = self.overlayList[0]
        self.defColor = (20, 150, 0)
        self.defThickness = 20
        self.brushThickness = self.defThickness
        self.xp, self.yp = 0, 0
        self.drawCanvas = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        self.clear_requested = False

    def latest_canvas_rgb(self):
        return cv2.cvtColor(self.drawCanvas, cv2.COLOR_BGR2RGB)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        img = cv2.flip(img, 1)
        img = cv2.GaussianBlur(img, (17, 17), cv2.BORDER_DEFAULT)

        if self.clear_requested:
            self.drawCanvas[:] = 0
            self.clear_requested = False

        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            x3, y3 = lmList[4][1:]
            x4, y4 = (x1 + x3) // 2, (y1 + y3) // 2

            fingers = self.detector.fingersUp()

            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                if y1 < HEADER_H:
                    if 0 < x1 < 183:
                        self.header = self.overlayList[0]
                        self.defColor = (20, 150, 0)
                        self.brushThickness = self.defThickness
                    elif 183 < x1 < 366:
                        self.header = self.overlayList[1]
                        self.defColor = (0, 50, 255)
                        self.brushThickness = self.defThickness
                    elif 366 < x1 < 549:
                        self.header = self.overlayList[2]
                        self.defColor = (210, 50, 0)
                        self.brushThickness = self.defThickness
                    elif 549 < x1 < 732:
                        self.header = self.overlayList[3]
                        self.defColor = (7, 210, 252)
                        self.brushThickness = self.defThickness
                    elif 732 < x1 < 915:
                        self.header = self.overlayList[4]
                        self.defColor = (102, 0, 255)
                        self.brushThickness = self.defThickness
                    elif 915 < x1 < 1098:
                        self.header = self.overlayList[5]
                        self.defColor = (255, 255, 255)
                        self.brushThickness = self.defThickness
                    elif x1 > 1098:
                        self.header = self.overlayList[6]
                        self.defColor = (0, 0, 0)
                        self.brushThickness = 80

            if fingers[1] and fingers[2] and fingers[3]:
                if x1 > 1098 and y1 < HEADER_H:
                    self.drawCanvas[:] = 0

            if fingers[1] and not fingers[2] and not fingers[0]:
                cv2.circle(img, (x1, y1), int(self.brushThickness // 2), self.defColor, cv2.FILLED)
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1
                cv2.line(img, (self.xp, self.yp), (x1, y1), self.defColor, self.brushThickness)
                cv2.line(self.drawCanvas, (self.xp, self.yp), (x1, y1), self.defColor, self.brushThickness)
                self.xp, self.yp = x1, y1

            if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                cv2.circle(img, (x1, y1), 15, (0, 0, 40), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 40), cv2.FILLED)
                cv2.line(img, (x1, y1), (x3, y3), (0, 0, 75), 10)
                length = math.hypot(x3 - x1, y3 - y1)
                self.brushThickness = max(1, int(length / 5))
                cv2.circle(img, (x4, y4), int(self.brushThickness // 1.25), self.defColor, cv2.FILLED)

        img[0:HEADER_H, 0:FRAME_W] = self.header
        gray = cv2.cvtColor(self.drawCanvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, inv)
        img = cv2.bitwise_or(img, self.drawCanvas)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Air Canvas", layout="wide")
st.title("Air Canvas")
st.caption("Draw in the air with your index finger. Two fingers up over the header to switch tools.")

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="air-canvas",
    video_processor_factory=AirCanvasProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear canvas", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.clear_requested = True

with col2:
    if ctx.video_processor and st.button("Capture drawing", use_container_width=True):
        rgb = ctx.video_processor.latest_canvas_rgb()
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if ok:
            st.download_button(
                "Download capture.png",
                data=buf.tobytes(),
                file_name="capture.png",
                mime="image/png",
                use_container_width=True,
            )

with st.expander("Gestures"):
    st.markdown(
        "- **Index + middle finger up** over the top bar: select a color or eraser.\n"
        "- **Index finger only**: draw.\n"
        "- **Thumb + index**: adjust brush thickness (distance between them).\n"
        "- **Index + middle + ring** in the eraser slot: clear the canvas."
    )
