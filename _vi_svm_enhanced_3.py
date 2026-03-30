"""
================================================================================
  VSI Visual Interface - Intelligence System
================================================================================
  1. Paper Scissors Stone game  — 3-sec countdown, score tracking, P-key shortcut
  2. Pose classifier            — 9 poses: Standing, Squat, Deep Squat,
                                           Push-up Up/Down, Arms Raised, T-Pose,
                                           Lunge, Sitting
  3. Exercise rep counter       — auto-counts push-ups AND squats via
                                  state-machine (up→down→up = 1 rep)
  4. Gesture temporal smoothing — deque of last 5 frames → mode vote,
                                  eliminates flicker
  5. Pose temporal smoothing    — deque of last 7 frames → stable label
  6. Joint angle overlay        — toggleable on-frame angle annotations
  7. Gesture map                — extended vocabulary
  8. ONNX emotion               — 
  9. Cosine guard on face recog — 

  Requirements:
  ─────────────────────────────────────────────────
      pip install PyQt5 | For building the graphical user interface (GUI).
      pip install opencv-contrib-python | For advanced image processing and additional OpenCV modules.
      pip install ultralytics | YOLOv8 face detection
      pip install facenet-pytorch | Face detection + embedding
      pip install mediapipe==0.10.20 | Hand, face, and body tracking
      pip install scikit-learn | SVM classifier for face recognition
      pip install joblib | Save/load trained models efficiently
      pip install "numpy<2.0" | For numerical computations and array manipulation.
                                Warning: numpy version 2.0 and above will fails in PYtorch
                                apps!
      pip install onnxruntime requests | Emotion Detection
      pip install omegaconf | 
      pip install pyside6 | Qt for python
      pip install torch | Deep learning backend
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      pip install icecream | For easy and advanced debugging/logging.
      pip install PyQt5Designer | For development of ui file. “pip install PyQt5 pyqt5-tools” is alternative to package 1 and 5
================================================================================
"""

import sys, os, time, threading, random, requests
from datetime import datetime
from pathlib import Path
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import torch

from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                              QFileDialog, QGraphicsScene, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QRectF

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp

try:
    import onnxruntime as ort
    ONNX_OK = True
except ImportError:
    ONNX_OK = False
    print("pip install onnxruntime requests  →  for emotion detection")

# ─────────────────────────────────────────────────────────────────────────────
UI_FILE          = "_vi_ocv_enhanced_2.1.ui"
COSINE_THRESHOLD = 0.65   # lower = more permissive face matching
EMOTION_INTERVAL = 10     # run emotion every N frames

ONNX_URL  = ("https://github.com/onnx/models/raw/main/validated/"
             "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")
ONNX_PATH = Path("OCV_data/emotion_ferplus.onnx")

EMOTION_LABELS = ["neutral 😐","happy 😁","surprise 😲","sad 😔","angry 😡","disgust 🤮","fear 😱","contempt 😏"]

EMOTION_BGR = {
    "happy":     (0, 220, 100), "sad":      (200, 100,  50),
    "angry":    (30,  30, 230), "surprise": (  0, 200, 255),
    "fear":     (160,  0, 200), "disgust":  ( 40, 160,  40),
    "neutral":  (180,180, 180), "contempt": (200, 200,   0),
}

# ── Extended gesture map ──────────────────────────────────────────────────────
# Key = (thumb, index, middle, ring, pinky)  1=extended
GESTURE_MAP = {
    (0,0,0,0,0): "Fist ✊",
    (1,1,1,1,1): "Open Hand 🖐",
    (0,1,1,0,0): "Peace ✌",
    (1,0,0,0,0): "Thumbs Up 👍",
    (0,0,0,0,1): "Pinky 🤙",
    (0,1,0,0,0): "Point ☝",
    (1,1,0,0,1): "Spider 🤟",
    (1,0,0,0,1): "Call Me 🤙",
    (1,1,1,0,0): "Three A",
    (0,1,1,1,0): "Three B",
    (0,1,1,1,1): "Four",
    (1,1,1,1,0): "Four (No Pinky)",
    (1,1,0,0,0): "Gun 👉",
    (0,0,1,0,0): "Middle 🖕",
    (0,0,0,1,0): "Ring",
    (1,0,1,0,0): "Two (Thumb+Mid)",
    (0,1,0,0,1): "Rock On 🤘",
    (0,0,1,1,1): "Okay 👌",
}

# ── RPS rules ─────────────────────────────────────────────────────────────────
RPS_CHOICES    = ["Stone", "Paper", "Scissors"]
RPS_EMOJI      = {"Stone": "✊", "Paper": "🖐", "Scissors": "✌"}
RPS_HAND_MAP   = {         # maps gesture states → RPS label
    (0,0,0,0,0): "Stone",
    (1,1,1,1,1): "Paper",
    (0,1,1,0,0): "Scissors",
}
RPS_WIN_PAIRS  = {("Stone","Scissors"), ("Paper","Stone"), ("Scissors","Paper")}

# ── Pose labels ───────────────────────────────────────────────────────────────
POSE_COLORS_BGR = {
    "Standing 🧍":     (0, 220,  80),
    "Squat":           (0, 200, 255),
    "Deep Squat":      (0, 150, 255),
    "Partial Squat":   (0, 180, 200),
    "Push-up (Up)":    (255, 180,  0),
    "Push-up (Down)":  (255,  80,  0),
    "Push-up":         (255, 130,  0),  # was missing entirely
    "Arms Raised 🙋‍♂️": (200,   0, 200),
    "T-Pose":          (200, 200,   0),
    "Lunge 🧎":        (200, 100,   0),
    "Sitting 🧘":      (150, 150, 255),
    "Unknown":         (180, 180, 180),
}


# ==============================================================================
#  Utility functions
# ==============================================================================
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def calc_angle(a, b, c) -> float:
    # Angle in degrees at landmark b, formed by a-b-c.
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    cos = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ==============================================================================
#  ONNX Emotion Detector 
# ==============================================================================
class OnnxEmotionDetector:
    def __init__(self):
        self.session = None
        self._download()
        self._load()

    def _download(self):
        if ONNX_PATH.exists():
            return
        ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading FERPlus emotion model (~33 MB) → {ONNX_PATH}")
        try:
            r = requests.get(ONNX_URL, stream=True, timeout=90)
            r.raise_for_status()
            with open(ONNX_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print("  Emotion model downloaded OK.")
        except Exception as e:
            print(f"  Download failed: {e}  — emotion detection unavailable this session.")

    def _load(self):
        if not ONNX_PATH.exists():
            return
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3   # suppress ONNX runtime info/warnings
            self.session = ort.InferenceSession(
                str(ONNX_PATH), sess_options=opts,
                providers=["CPUExecutionProvider"])
            print("  ONNX emotion model loaded.")
        except Exception as e:
            print(f"  ONNX load error: {e}")

    def predict(self, face_bgr: np.ndarray) -> tuple:
        """
        Return (label, confidence_0_to_1).

        Accuracy improvements vs naive resize→infer:
          1. CLAHE  — equalises contrast region-by-region, handles shadows/bright
          2. Gamma  — mild brightening (gamma=0.9) lifts dark faces
          3. Multi-crop ensemble — centre + 4 corners, average softmax probs
             (same trick used in ImageNet benchmarks, ~2-3% accuracy gain)
        """
        if self.session is None:
            return "unavailable", 0.0
        try:
            # --- Pre-process ---
            grey = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            grey = cv2.resize(grey, (72, 72)) # slightly bigger for crops

            # CLAHE: adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            grey  = clahe.apply(grey)

            # Gamma correction (gamma < 1 brightens)
            lut   = np.array([((i / 255.0) ** 0.9) * 255
                              for i in range(256)], dtype=np.uint8)
            grey  = cv2.LUT(grey, lut)

            # --- 5-crop ensemble (centre + 4 corners, each 64x64) ---
            crops = [
                grey[4:68, 4:68],    # centre
                grey[0:64, 0:64],    # top-left
                grey[0:64, 8:72],    # top-right
                grey[8:72, 0:64],    # bottom-left
                grey[8:72, 8:72],    # bottom-right
            ]
            avg_prob = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
            for crop in crops:
                inp  = crop.astype(np.float32).reshape(1, 1, 64, 64)
                out  = self.session.run(None, {"Input3": inp})[0]
                avg_prob += _softmax(out[0])
            avg_prob /= len(crops)

            idx = int(np.argmax(avg_prob))
            return EMOTION_LABELS[idx], float(avg_prob[idx])
        except Exception:
            return "error", 0.0


# ==============================================================================
#  Pose Classifier  (rule-based, uses MediaPipe joint angles)
# ==============================================================================
def classify_pose(lm) -> tuple:
    """
    lm = list of MediaPipe PoseLandmark objects (33 total)
    Returns (pose_label: str, angles: dict)
    """
    # Indices: shoulders 11/12, elbows 13/14, wrists 15/16
    #          hips 23/24, knees 25/26, ankles 27/28
    try:
        ls, rs = lm[11], lm[12]
        le, re = lm[13], lm[14]
        lw, rw = lm[15], lm[16]
        lh, rh = lm[23], lm[24]
        lk, rk = lm[25], lm[26]
        la, ra = lm[27], lm[28]

        knee_l  = calc_angle(lh, lk, la)
        knee_r  = calc_angle(rh, rk, ra)
        elbow_l = calc_angle(ls, le, lw)
        elbow_r = calc_angle(rs, re, rw)
        hip_l   = calc_angle(ls, lh, lk)
        hip_r   = calc_angle(rs, rh, rk)

        avg_knee  = (knee_l  + knee_r)  / 2
        avg_elbow = (elbow_l + elbow_r) / 2
        avg_hip   = (hip_l   + hip_r)   / 2

        angles = {"knee": avg_knee, "elbow": avg_elbow, "hip": avg_hip}

        # Key y-positions  (y increases downward in normalised coords)
        shoulder_y = (ls.y + rs.y) / 2
        hip_y      = (lh.y + rh.y) / 2
        knee_y     = (lk.y + rk.y) / 2
        ankle_y    = (la.y + ra.y) / 2
        wrist_y    = (lw.y + rw.y) / 2
        wrist_xspread  = abs(lw.x - rw.x)
        shld_xspread   = abs(ls.x - rs.x)

        body_vertical   = (shoulder_y < hip_y) and (hip_y < ankle_y)
        body_horizontal = abs(shoulder_y - ankle_y) < 0.30

        # ── Push-up: body roughly horizontal ──────────────────────────────────
        if body_horizontal:
            if avg_elbow > 145:
                return "Push-up (Up)", angles
            elif avg_elbow < 90:
                return "Push-up (Down)", angles
            else:
                return "Push-up", angles

        # ── Upright poses ──────────────────────────────────────────────────────
        if body_vertical:
            # T-Pose: arms spread wide horizontally
            if (wrist_xspread > shld_xspread * 1.6
                    and abs(lw.y - ls.y) < 0.12
                    and abs(rw.y - rs.y) < 0.12):
                return "T-Pose", angles

            # Arms raised (wrists above shoulders)
            if wrist_y < shoulder_y - 0.08 and avg_knee > 145:
                return "Arms Raised 🙋‍♂️", angles

            # Sitting: hips close to knee height
            if abs(hip_y - knee_y) < 0.12 and avg_knee < 130:
                return "Sitting 🧘", angles

            # Lunge: large difference between left and right knee angles
            if abs(knee_l - knee_r) > 45:
                return "Lunge 🧎", angles

            # Squat / standing by knee angle
            if avg_knee > 165:
                return "Standing 🧍", angles
            elif avg_knee > 140:
                return "Partial Squat", angles
            elif avg_knee > 100:
                return "Squat", angles
            else:
                return "Deep Squat", angles

        return "Unknown", angles

    except Exception:
        return "Unknown", {}


# ==============================================================================
#  Exercise Rep Counter  (state machine)
# ==============================================================================
class ExerciseCounter:
    """Counts push-up and squat reps via pose state transitions."""

    def __init__(self):
        self.squat_state  = "up"    # "up" | "down"
        self.pushup_state = "up"
        self.squat_reps   = 0
        self.pushup_reps  = 0

    def update(self, pose: str) -> tuple:
        # Squat: standing → squat/deep squat → standing = 1 rep (ignore partial)
        if pose in ("Squat", "Deep Squat"):        # catches "Squat", "Deep Squat"
            if self.squat_state == "up":
               self.squat_state = "down"
        elif "Standing" in pose:                   # catches "Standing 🧍" ← this was the main bug
            if self.squat_state == "down":
                self.squat_state = "up"
                self.squat_reps += 1

        # Push-up: up → down → up = 1 rep
        if "Down" in pose and "Push" in pose:      # catches "Push-up (Down)"
            if self.pushup_state == "up":
                self.pushup_state = "down"
        elif "Up" in pose and "Push" in pose:      # catches "Push-up (Up)"
            if self.pushup_state == "down":
                self.pushup_state = "up"
                self.pushup_reps += 1

        return self.squat_reps, self.pushup_reps

    def reset(self):
        self.squat_state  = "up"
        self.pushup_state = "up"
        self.squat_reps   = 0
        self.pushup_reps  = 0


# ==============================================================================
#  Main Application
# ==============================================================================
class FaceTrainerApp(QMainWindow):

    def __init__(self):
        super().__init__()

        if not os.path.exists(UI_FILE):
            print(f"CRITICAL: UI file '{UI_FILE}' not found in {os.getcwd()}")
            sys.exit(1)
        uic.loadUi(UI_FILE, self)
        self.statusBar().showMessage("Initialising models…")

        # QGraphicsView
        self.graphics_view = self.findChild(QtWidgets.QGraphicsView, "graphicsView")
        if not self.graphics_view:
            print("CRITICAL: 'graphicsView' not found in UI.")
            sys.exit(1)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.pixmap_item = None

        # ── Runtime state ──────────────────────────────────────────────────────
        self.current_frame        = None
        self.confidence_threshold = 50.0
        self.frame_count          = 0
        self.fps_start_time       = time.time()
        self.current_fps          = 0.0

        # Emotion — temporal smoothing over last 8 predictions
        self.last_emotion      = "neutral"
        self.last_emotion_conf = 0.0
        self.emotion_analyzing = False
        self.show_emotion      = True
        self.emotion_history   = deque(maxlen=8)   # (label, conf) pairs for voting

        # Pose
        self.show_pose         = False
        self.show_angles       = False
        self.pose_history      = deque(maxlen=7)   # temporal smoother

        # Gesture smoother
        self.gesture_history   = deque(maxlen=5)
        self.current_hand_states: tuple = (0,0,0,0,0)  # for RPS capture

        # Exercise counter
        self.exercise_counter  = ExerciseCounter()
        self.count_reps        = True

        # RPS game
        self.rps_wins    = 0
        self.rps_losses  = 0
        self.rps_draws   = 0
        self.rps_timer   = QTimer()
        self.rps_timer.timeout.connect(self._rps_tick)
        self.rps_countdown = 0

        # Known embeddings cache for cosine guard
        self.known_embeddings: dict = {}

        os.makedirs("snapshots", exist_ok=True)
        os.makedirs("OCV_data",  exist_ok=True)

        self._apply_stylesheet()
        self._connect_signals()

        # ── Load models ────────────────────────────────────────────────────────
        self.log("Loading YOLOv8 face detector…")
        self.yolo_model = YOLO("yolov8n-face.pt")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lbl_device.setText(self.device.upper())
        self.log(f"Device: {self.device.upper()}")

        self.log("Loading FaceNet (InceptionResnetV1 vggface2)…")
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        self.log("Loading MediaPipe Hands (complexity=1)…")
        self._mp_h  = mp.solutions.hands
        self.hands_det = self._mp_h.Hands(
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
            max_num_hands=2,
        )

        self.log("Loading MediaPipe Pose (complexity=1)…")
        self._mp_p  = mp.solutions.pose
        self.pose_det = self._mp_p.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            smooth_landmarks=True,
        )
        self._mp_draw = mp.solutions.drawing_utils

        self.log("Loading ONNX emotion model (FERPlus-8)... ...")
        if ONNX_OK:
            self.emotion_det = OnnxEmotionDetector()
        else:
            self.emotion_det = None
            self.log("⚠  onnxruntime missing — pip install onnxruntime requests")

        self.svm_model = None
        self.cap       = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._update_frame)

        emo_ok = self.emotion_det and self.emotion_det.session is not None
        self.statusBar().showMessage(
            f"Ready | {self.device.upper()} | "
            f"Emotion: {'ONNX OK' if emo_ok else 'unavailable'} | "
            "Load SVM → Start Camera")
        self.log("=" * 52)
        self.log("All models ready.  Load SVM then start camera.")
        self.log("RPS: press P to play.  Pose: enable skeleton overlay.")
        self.log("=" * 52)

    # ══════════════════════════════════════════════════════════════════════════
    #  STYLESHEET
    # ══════════════════════════════════════════════════════════════════════════
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 4px;
                color: #89b4fa;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover   { background-color: #45475a; }
            QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
            QPushButton:disabled{ background-color: #1e1e2e; color: #585b70; }
            QLabel { color: #cdd6f4; }
            QTextEdit {
                background-color: #181825;
                color: #a6e3a1;
                border: 1px solid #45475a;
                border-radius: 4px;
                font-family: Consolas, Courier New, monospace;
                font-size: 12px;
            }
            QSlider::groove:horizontal { height:6px; background:#45475a; border-radius:3px; }
            QSlider::handle:horizontal {
                background:#89b4fa; border:1px solid #45475a;
                width:14px; margin:-4px 0; border-radius:7px;
            }
            QSlider::sub-page:horizontal { background:#89b4fa; border-radius:3px; }
            QCheckBox { spacing:8px; }
            QCheckBox::indicator {
                width:16px; height:16px;
                border:1px solid #45475a; border-radius:3px; background:#313244;
            }
            QCheckBox::indicator:checked { background:#89b4fa; }
            QMenuBar  { background-color:#181825; color:#cdd6f4; }
            QMenuBar::item:selected { background:#313244; }
            QMenu { background-color:#1e1e2e; color:#cdd6f4; border:1px solid #45475a; }
            QMenu::item:selected { background-color:#313244; }
            QStatusBar { background-color:#181825; color:#a6adc8; }
            QGraphicsView {
                border:2px solid #45475a; border-radius:6px; background-color:#11111b;
            }
        """)

    # ══════════════════════════════════════════════════════════════════════════
    #  SIGNAL CONNECTIONS
    # ══════════════════════════════════════════════════════════════════════════
    def _connect_signals(self):
        self.actionCamera_SVM.triggered.connect(self.start_camera)
        self.actionCamera_SVM_off.triggered.connect(self.stop_camera)
        self.actionLoad_SVM.triggered.connect(self.load_svm)
        self.actionTrain_SVM.triggered.connect(self._train_info)
        self.actionSnapshot.triggered.connect(self.save_snapshot)
        self.actionRPS_Play.triggered.connect(self.start_rps)
        self.actionReset_Reps.triggered.connect(self._reset_reps)
        self.actionExit.triggered.connect(self.close)
        self.actionToggleEmotion.triggered.connect(self._menu_emo)
        self.actionTogglePose.triggered.connect(self._menu_pose)
        self.actionToggleAngles.triggered.connect(self._menu_angles)
        self.actionAbout.triggered.connect(self._about)

        self.btn_camera_on.clicked.connect(self.start_camera)
        self.btn_camera_off.clicked.connect(self.stop_camera)
        self.btn_load_svm.clicked.connect(self.load_svm)
        self.btn_snapshot.clicked.connect(self.save_snapshot)
        self.btn_clear_log.clicked.connect(self.my_terminal.clear)
        self.btn_rps_play.clicked.connect(self.start_rps)
        self.btn_rps_reset.clicked.connect(self._reset_rps_score)
        self.btn_reset_reps.clicked.connect(self._reset_reps)

        self.slider_conf.valueChanged.connect(self._on_slider)
        self.chk_emotion.stateChanged.connect(self._chk_emo)
        self.chk_pose.stateChanged.connect(self._chk_pose)
        self.chk_count_reps.stateChanged.connect(self._chk_reps)

    # ── Logging ───────────────────────────────────────────────────────────────
    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.my_terminal.append(f"[{ts}]  {msg}")

    def _lbl(self, n): return self.findChild(QtWidgets.QLabel, n)

    # ── Slider / checkbox callbacks ───────────────────────────────────────────
    def _on_slider(self, v):
        self.confidence_threshold = float(v)
        self.lbl_threshold_val.setText(f"{v}%")

    def _chk_emo(self, s):
        self.show_emotion = (s == Qt.Checked)
        self.actionToggleEmotion.setChecked(self.show_emotion)

    def _chk_pose(self, s):
        self.show_pose = (s == Qt.Checked)
        self.actionTogglePose.setChecked(self.show_pose)
        if not self.show_pose:
            self.lbl_pose_name.setText("—  (enable Pose Skeleton)")
            self.lbl_knee_angle.setText("—")
            self.lbl_elbow_angle.setText("—")
            self.lbl_hip_angle.setText("—")

    def _chk_reps(self, s):
        self.count_reps = (s == Qt.Checked)

    def _menu_emo(self):
        self.show_emotion = self.actionToggleEmotion.isChecked()
        self.chk_emotion.setChecked(self.show_emotion)

    def _menu_pose(self):
        self.show_pose = self.actionTogglePose.isChecked()
        self.chk_pose.setChecked(self.show_pose)

    def _menu_angles(self):
        self.show_angles = self.actionToggleAngles.isChecked()
        self.log(f"Angle overlay {'ON' if self.show_angles else 'OFF'}")

    def _train_info(self):
        QMessageBox.information(self, "Train SVM",
            "Run in your terminal (venv active):\n\n"
            "    python _vi_svm_train_enhanced.py\n\n"
            "Edit TRAIN_DIR and PEOPLE inside that script, then\n"
            "load the [ .pkl ] via Button [Load SVM Model].")

    def _about(self):
        QMessageBox.about(self, "VSI Manual",
            "<b>VSI Visual Interface - Intelligence System  </b><br><br>"
            "<b>Pipeline:</b><br>"
            "Face Detection — YOLOv8<br>"
            "Face Recognition — FaceNet + cosine guard + SVM<br>"
            "Emotion — FERPlus ONNX<br>"
            "Gesture — MediaPipe Hands + gesture map<br>"
            "Pose — MediaPipe Pose + rule-based classifier<br>"
            "Exercise — state-machine rep counter<br>"
            "RPS Game — countdown + score<br><br>"
            "<b>Shortcuts:</b><br>"
            "Ctrl+R Camera ON | Ctrl+T Camera OFF<br>"
            "Ctrl+S Snapshot  | P Play RPS | Ctrl+Q Exit")

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMERA
    # ══════════════════════════════════════════════════════════════════════════
    def start_camera(self):
        if self.frame_timer.isActive():
            self.statusBar().showMessage("Camera already running.")
            return
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.fps_start_time = time.time()
            self.frame_count    = 0
            self.frame_timer.start(30)
            self.log("Camera started (index 0).")
        else:
            self.statusBar().showMessage("ERROR: Cannot open camera (try index 1)")
            self.log("ERROR: cv2.VideoCapture(0) failed.")
            self.cap = None

    def stop_camera(self):
        if self.frame_timer.isActive():
            self.frame_timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.pixmap_item = None
        self.scene.clear()
        self.statusBar().showMessage("Camera stopped.")
        self.log("Camera stopped.")

    def save_snapshot(self):
        if self.current_frame is None:
            self.log("No frame — start camera first.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"snapshots/snap_{ts}.jpg"
        cv2.imwrite(path, self.current_frame)
        self.statusBar().showMessage(f"Snapshot saved → {path}")
        self.log(f"Snapshot: {path}")

    def load_svm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SVM (.pkl)", "", "PKL Files (*.pkl)")
        if not path:
            return
        try:
            self.svm_model = joblib.load(path)
            classes = list(self.svm_model.classes_)
            self.lbl_model_status.setText(
                f"{os.path.basename(path)}  [{', '.join(classes)}]")
            self.log(f"SVM loaded: {path}")
            self.log(f"Identities: {classes}")
            cache = path.replace(".pkl", "_embeddings.npy")
            if os.path.exists(cache):
                self.known_embeddings = np.load(cache, allow_pickle=True).item()
                self.log(f"Embedding cache loaded ({len(self.known_embeddings)} people).")
            else:
                self.log("No embedding cache — cosine guard disabled.")
                self.log("  Tip: run _vi_svm_train_enhanced.py to create cache.")
        except Exception as e:
            self.log(f"ERROR loading SVM: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  RPS GAME
    # ══════════════════════════════════════════════════════════════════════════
    def start_rps(self):
        if self.rps_timer.isActive():
            return
        if not self.frame_timer.isActive():
            self.log("Start the camera first before playing RPS.")
            return
        self.rps_countdown = 3
        self.btn_rps_play.setEnabled(False)
        self.lbl_rps_status.setText("3...")
        self.lbl_rps_result.setText("—")
        self.lbl_player_choice.setText("—")
        self.lbl_computer_choice.setText("—")
        self.rps_timer.start(1000)
        self.log("RPS: countdown started — get your hand ready!")

    def _rps_tick(self):
        self.rps_countdown -= 1
        if self.rps_countdown > 0:
            self.lbl_rps_status.setText(f"{self.rps_countdown}...")
        else:
            self.rps_timer.stop()
            self.lbl_rps_status.setText("SHOW! 👀")
            QTimer.singleShot(400, self._rps_capture)

    def _rps_capture(self):
        """Capture current hand states and resolve the round."""
        states   = self.current_hand_states
        player   = RPS_HAND_MAP.get(states)
        computer = random.choice(RPS_CHOICES)

        if player is None:
            self.lbl_rps_status.setText("Invalid ❌")
            self.log(f"RPS: invalid gesture {states} — show Fist/Open/Peace")
            self.btn_rps_play.setEnabled(True)
            return

        # Determine winner
        if player == computer:
            result, result_color = "Draw! 🤝", "#cba6f7"
            self.rps_draws += 1
        elif (player, computer) in RPS_WIN_PAIRS:
            result, result_color = "You WIN! 🎉", "#a6e3a1"
            self.rps_wins += 1
        else:
            result, result_color = "You LOSE 😞", "#f38ba8"
            self.rps_losses += 1

        self.lbl_player_choice.setText(f"{RPS_EMOJI[player]} {player}")
        self.lbl_computer_choice.setText(f"{RPS_EMOJI[computer]} {computer}")
        self.lbl_rps_result.setText(result)
        self.lbl_rps_result.setStyleSheet(
            f"color: {result_color}; font-size: 14px; font-weight: bold;")
        self.lbl_rps_score.setText(
            f"W: {self.rps_wins}   L: {self.rps_losses}   D: {self.rps_draws}")
        self.lbl_rps_status.setText("Done!")
        self.btn_rps_play.setEnabled(True)
        self.log(f"RPS: You={player}  CPU={computer}  → {result}")

    def _reset_rps_score(self):
        self.rps_wins = self.rps_losses = self.rps_draws = 0
        self.lbl_rps_score.setText("W: 0   L: 0   D: 0")
        self.lbl_rps_result.setText("—")
        self.lbl_rps_result.setStyleSheet("")
        self.lbl_player_choice.setText("—")
        self.lbl_computer_choice.setText("—")
        self.lbl_rps_status.setText("Ready")
        self.log("RPS score reset.")

    # ══════════════════════════════════════════════════════════════════════════
    #  EXERCISE COUNTER
    # ══════════════════════════════════════════════════════════════════════════
    def _reset_reps(self):
        self.exercise_counter.reset()
        self.lbl_rep_count.setText("0")
        self.lbl_exercise_name.setText("")
        self.log("Rep counter reset.")

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN FRAME LOOP
    # ══════════════════════════════════════════════════════════════════════════
    def _update_frame(self):
        if not self.cap:
            self.frame_timer.stop()
            return
        ret, frame = self.cap.read()
        if not ret:
            self.log("Frame read failed — stopping camera.")
            self.stop_camera()
            return

        self.current_frame  = frame.copy()
        self.frame_count   += 1

        # FPS
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps    = self.frame_count / elapsed
            self.frame_count    = 0
            self.fps_start_time = time.time()
            self.lbl_fps.setText(f"{self.current_fps:.1f}")

        # ── 1. Face detection + recognition ──────────────────────────────────
        boxes = self._yolo_detect(frame)
        self.lbl_face_count.setText(str(len(boxes)))

        best_name, best_conf = "—", 0.0
        for x1, y1, x2, y2 in boxes:
            pad = 12
            crop = frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                         max(0,x1-pad):min(frame.shape[1],x2+pad)]
            if crop.size == 0:
                continue
            name, conf = self._recognize_face(crop)
            confident  = conf >= self.confidence_threshold
            color      = (0, 220, 80) if confident else (0, 60, 220)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{name}  {conf:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            if conf > best_conf:
                best_conf, best_name = conf, name

            # Emotion overlay
            if self.show_emotion and self.emotion_det:
                emo = self.last_emotion
                ec  = EMOTION_BGR.get(emo.lower(), (180,180,180))
                cv2.putText(frame, f"{emo}  {self.last_emotion_conf*100:.0f}%",
                            (x1, y2+22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, ec, 2, cv2.LINE_AA)
                if (not self.emotion_analyzing
                        and self.frame_count % EMOTION_INTERVAL == 0):
                    self._async_emotion(crop.copy())

        self.lbl_face_name.setText(best_name)
        self.lbl_conf_val.setText(f"{best_conf:.1f}%" if best_conf > 0 else "—")
        self.lbl_emotion.setText(
            f"{self.last_emotion}  ({self.last_emotion_conf*100:.0f}%)"
            if self.show_emotion else "disabled")

        # ── 2. Pose skeleton + classifier ─────────────────────────────────────
        if self.show_pose:
            rgb_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pres = self.pose_det.process(rgb_pose)
            if pres.pose_landmarks:
                self._mp_draw.draw_landmarks(
                    frame, pres.pose_landmarks, self._mp_p.POSE_CONNECTIONS,
                    self._mp_draw.DrawingSpec((0,255,200), 2, 3),
                    self._mp_draw.DrawingSpec((0,150,255), 2),
                )
                lm = pres.pose_landmarks.landmark
                pose_raw, angles = classify_pose(lm)
                self.pose_history.append(pose_raw)
                pose = Counter(self.pose_history).most_common(1)[0][0]

                pose_color = POSE_COLORS_BGR.get(pose, (180,180,180))
                cv2.putText(frame, pose, (10, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, pose_color, 2, cv2.LINE_AA)

                # Angle overlays
                if self.show_angles and angles:
                    h, w = frame.shape[:2]
                    def lm_px(idx): return (int(lm[idx].x*w), int(lm[idx].y*h))
                    for idx, key, label in [
                        (26, "knee",  "K"), (14, "elbow", "E"), (24, "hip", "H")]:
                        px = lm_px(idx)
                        val = angles.get(key, 0)
                        cv2.putText(frame, f"{label}:{val:.0f}",
                                    (px[0]+8, px[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (200,200,0), 1, cv2.LINE_AA)

                # Update right panel
                self.lbl_pose_name.setText(pose)
                self.lbl_knee_angle.setText(f"{angles.get('knee',0):.1f}°")
                self.lbl_elbow_angle.setText(f"{angles.get('elbow',0):.1f}°")
                self.lbl_hip_angle.setText(f"{angles.get('hip',0):.1f}°")

                # Rep counting
                if self.count_reps:
                    sq, pu = self.exercise_counter.update(pose)
                    total = sq + pu
                    self.lbl_rep_count.setText(str(total))
                    #if pose in ("Squat","Deep Squat","Partial Squat","Push-up (Down)","Push-up (Up)"):
                    #    ex = "Squats" if "Squat" in pose else "Push-ups"
                    #    self.lbl_exercise_name.setText(f"{ex}  (sq:{sq}  pu:{pu})")
                    
                    # Always update counts display
                    self.lbl_exercise_name.setText(f"sq:{sq}  pu:{pu}")

                    # Optional: show current active exercise
                    if "Squat" in pose:
                        self.lbl_exercise_name.setText(f"Squats  (sq:{sq}  pu:{pu})")
                    elif "Push" in pose:
                        self.lbl_exercise_name.setText(f"Push-ups  (sq:{sq}  pu:{pu})")
                        
        # ── 3. Hand gesture ───────────────────────────────────────────────────
        rgb_hand = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hres     = self.hands_det.process(rgb_hand)
        total_fingers = 0
        gesture_texts = []
        rps_candidates = []

        if hres.multi_hand_landmarks:
            for hlm, hside in zip(hres.multi_hand_landmarks, hres.multi_handedness):
                self._mp_draw.draw_landmarks(
                    frame, hlm, self._mp_h.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec((255,100,0), 2, 4),
                    self._mp_draw.DrawingSpec((255,230,0), 2),
                )
                states, n = self._finger_states(hlm, hside)
                total_fingers += n

                # Temporal smoothing per hand (shared history — works fine for 1-2 hands)
                self.gesture_history.append(states)
                smoothed = Counter(self.gesture_history).most_common(1)[0][0]
                self.current_hand_states = smoothed   # for RPS capture

                gesture = self._classify_gesture(smoothed)
                gesture_texts.append(gesture)

                rps = RPS_HAND_MAP.get(smoothed)
                if rps:
                    rps_candidates.append(rps)

                # Per-hand label
                H, W, _ = frame.shape
                wrist = hlm.landmark[self._mp_h.HandLandmark.WRIST]
                wx, wy = int(wrist.x*W), max(30, int(wrist.y*H)-30)
                cv2.putText(frame, gesture, (wx, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,220,255), 2, cv2.LINE_AA)

        self.lbl_fingers.setText(str(total_fingers))
        self.lbl_gesture.setText(", ".join(gesture_texts) if gesture_texts else "—")
        self.lbl_rps_cat.setText(", ".join(rps_candidates) if rps_candidates else "—")

        # Status bar
        self.statusBar().showMessage(
            f"FPS: {self.current_fps:.1f}  |  "
            f"Faces: {len(boxes)}  |  "
            f"Gesture: {', '.join(gesture_texts) or 'none'}  |  "
            f"Threshold: {int(self.confidence_threshold)}%")

        # ── 4. Render ─────────────────────────────────────────────────────────
        disp  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = disp.shape
        qimg   = QtGui.QImage(disp.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.graphics_view.fitInView(
                self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

    # ══════════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def _yolo_detect(self, frame) -> list:
        res = self.yolo_model(frame, verbose=False)
        if not res or res[0].boxes is None:
            return []
        return [[int(v) for v in box[:4]]
                for box in res[0].boxes.xyxy.cpu().numpy()]

    # ── Face recognition ──────────────────────────────────────────────────────
    def _get_embedding(self, face_bgr):
        try:
            rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            resz = cv2.resize(rgb, (160, 160)).astype(np.float32) / 255.0
            t    = torch.tensor(resz).permute(2,0,1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.resnet(t).squeeze(0).cpu().numpy()
            return emb / (np.linalg.norm(emb) + 1e-9)
        except Exception:
            return None

    def _cosine_ok(self, name, emb) -> bool:
        if name not in self.known_embeddings:
            return True
        return float(np.dot(emb, self.known_embeddings[name])) >= COSINE_THRESHOLD

    def _recognize_face(self, face_bgr) -> tuple:
        if self.svm_model is None:
            return "Load SVM", 0.0
        emb = self._get_embedding(face_bgr)
        if emb is None:
            return "Error", 0.0
        probs = self.svm_model.predict_proba(emb.reshape(1,-1))[0]
        idx   = int(np.argmax(probs))
        name  = str(self.svm_model.classes_[idx])
        conf  = float(probs[idx]) * 100.0
        if not self._cosine_ok(name, emb):
            return "Unknown", conf * 0.4
        return name, conf

    # ── Finger states ─────────────────────────────────────────────────────────
    def _finger_states(self, hlm, hside) -> tuple:
        H     = self._mp_h
        lm    = hlm.landmark
        label = hside.classification[0].label

        TIP = [H.HandLandmark.THUMB_TIP,    H.HandLandmark.INDEX_FINGER_TIP,
               H.HandLandmark.MIDDLE_FINGER_TIP, H.HandLandmark.RING_FINGER_TIP,
               H.HandLandmark.PINKY_TIP]
        PIP = [H.HandLandmark.THUMB_IP,     H.HandLandmark.INDEX_FINGER_PIP,
               H.HandLandmark.MIDDLE_FINGER_PIP, H.HandLandmark.RING_FINGER_PIP,
               H.HandLandmark.PINKY_PIP]
        MCP = [H.HandLandmark.THUMB_CMC,    H.HandLandmark.INDEX_FINGER_MCP,
               H.HandLandmark.MIDDLE_FINGER_MCP, H.HandLandmark.RING_FINGER_MCP,
               H.HandLandmark.PINKY_MCP]

        states = []
        # Thumb: x-axis, side-aware
        tx, cx = lm[TIP[0]].x, lm[MCP[0]].x
        states.append(1 if (label=="Right" and tx>cx) or (label=="Left" and tx<cx) else 0)
        # Fingers 1-4: tip above both pip and mcp
        for i in range(1, 5):
            states.append(1 if (lm[TIP[i]].y < lm[PIP[i]].y and
                                lm[TIP[i]].y < lm[MCP[i]].y) else 0)
        t = tuple(states)
        return t, sum(t)

    # ── Gesture classification ────────────────────────────────────────────────
    def _classify_gesture(self, states: tuple) -> str:
        name = GESTURE_MAP.get(states)
        if name:
            return name
        n = sum(states)
        return f"{n} finger{'s' if n!=1 else ''}"

    # ── Async emotion inference ───────────────────────────────────────────────
    def _async_emotion(self, face_bgr):
        """
        Runs ONNX emotion inference on a daemon thread.
        Results are accumulated in self.emotion_history (deque) and a
        weighted vote picks the final label — this smooths out single-frame
        mis-classifications, boosting practical accuracy significantly.
        """
        def _worker():
            try:
                label, conf = self.emotion_det.predict(face_bgr)
                if label not in ("unavailable", "error"):
                    self.emotion_history.append((label.capitalize(), conf))

                    # Weighted vote: sum confidence per label, pick highest
                    scores: dict = {}
                    for lbl, c in self.emotion_history:
                        scores[lbl] = scores.get(lbl, 0.0) + c
                    best_label = max(scores, key=scores.get)
                    # Average confidence for the winning label
                    winning_confs = [c for l, c in self.emotion_history if l == best_label]
                    best_conf     = sum(winning_confs) / len(winning_confs)

                    self.last_emotion      = best_label
                    self.last_emotion_conf = best_conf
            except Exception:
                pass
            finally:
                self.emotion_analyzing = False
        self.emotion_analyzing = True
        threading.Thread(target=_worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ══════════════════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        self.stop_camera()
        self.hands_det.close()
        self.pose_det.close()
        event.accept()


# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceTrainerApp()
    window.show()
    sys.exit(app.exec_())
