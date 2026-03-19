from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing import image
import pickle
from pathlib import Path

DEFAULT_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "nothing", "space", "del",
]

def preprocess_pil(pil_img: Image.Image, input_size=(224, 224)) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    img = pil_img.resize(input_size)
    img_array = image.img_to_array(img).astype("float32")
    return np.expand_dims(img_array, axis=0)


def load_class_names(pkl_path: str = "class_names.pkl") -> list[str]:
    p = Path(pkl_path)
    if not p.exists():
        return DEFAULT_LABELS
    try:
        with p.open("rb") as f:
            class_names = pickle.load(f)
        if isinstance(class_names, (list, tuple)) and len(class_names) > 0:
            return list(class_names)
    except Exception:
        pass
    return DEFAULT_LABELS

def _softmax_topk(preds: np.ndarray, labels: list[str], k: int = 3) -> list[tuple[str, float]]:
    if preds.ndim == 2:
        preds = preds[0]
    idx = np.argsort(preds)[-k:][::-1]
    return [(labels[i], float(preds[i])) for i in idx]


def _hand_bbox_square_from_landmarks(
    hand_landmarks,
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.25,
) -> tuple[int, int, int, int] | None:
    # MediaPipe landmarks are normalized coordinates in [0,1].
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    if not xs or not ys:
        return None

    x_min = min(xs) * img_w
    x_max = max(xs) * img_w
    y_min = min(ys) * img_h
    y_max = max(ys) * img_h

    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    size = max(box_w, box_h)
    size = size * (1.0 + float(margin_ratio))  # expand for context

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x1 = int(max(0, cx - size / 2.0))
    y1 = int(max(0, cy - size / 2.0))
    x2 = int(min(img_w, x1 + size))
    y2 = int(min(img_h, y1 + size))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _skin_bbox_square_from_bgr(
    img_bgr: np.ndarray,
    margin_ratio: float = 0.25,
    min_area_ratio: float = 0.01,
) -> tuple[int, int, int, int] | None:
    """
    Fallback hand ROI when MediaPipe is not available.
    Detect skin-like pixels and take the largest contour bbox.
    """
    import cv2  # local import to keep startup light

    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    # Common skin color range in YCrCb (works reasonably for many webcams).
    lower = (0, 133, 77)
    upper = (255, 173, 127)
    mask = cv2.inRange(ycrcb, lower, upper)

    # Cleanup noise and fill holes a bit.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(min_area_ratio) * float(h * w):
        return None

    x, y, bw, bh = cv2.boundingRect(contour)
    if bw <= 1 or bh <= 1:
        return None

    box_w = float(bw)
    box_h = float(bh)
    size = max(box_w, box_h)
    size = size * (1.0 + float(margin_ratio))

    cx = x + bw / 2.0
    cy = y + bh / 2.0

    x1 = int(max(0, cx - size / 2.0))
    y1 = int(max(0, cy - size / 2.0))
    x2 = int(min(w, x1 + size))
    y2 = int(min(h, y1 + size))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _render_prediction(model, labels: list[str], pil_image: Image.Image) -> tuple[str, float, list[tuple[str, float]]]:
    x = preprocess_pil(pil_image, input_size=(224, 224))
    preds = model.predict(x, verbose=0)
    pred_idx = int(np.argmax(preds))
    label = labels[pred_idx]
    conf = float(np.max(preds))
    top3 = _softmax_topk(preds, labels, k=min(3, len(labels)))
    return label, conf, top3


def main():
    @st.cache_resource
    def load_asl_model(model_path: str):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    st.title("American Sign Language (ASL) Prediction App")

    model_candidates = [
        "best_asl_model.h5",
        "asl_model_final.h5",
        "epoch_10_best_asl_model.h5",
        "epoch_9_best_asl_model.h5",
    ]
    existing_models = [p for p in model_candidates if Path(p).exists()]
    default_model = existing_models[0] if existing_models else model_candidates[0]

    model_path = st.sidebar.selectbox("Model to use", model_candidates, index=model_candidates.index(default_model))
    labels = load_class_names("class_names.pkl")
    model = load_asl_model(model_path)
    if model is None:
        st.stop()

    option = st.selectbox("Choose input type", ("Upload Image", "Use Webcam"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Uploaded Image.", use_column_width=True)  # PIL object
            if st.button("Predict"):
                predicted_label, confidence, top3 = _render_prediction(model, labels, pil_image)
                st.write(f"**Prediction:** {predicted_label} with {confidence*100:.2f}% confidence.")
                st.write("**Top-3:**")
                st.write({k: f"{v*100:.2f}%" for k, v in top3})
                
    elif option == "Use Webcam":
        st.write("Webcam prediction (snapshot).")
        st.write("Nếu bạn muốn **real-time (live video)**, hãy chọn mục **Real-time Webcam** bên dưới.")

        picture = st.camera_input("Chụp một ảnh từ webcam")

        if picture is not None:
            cam_image = Image.open(picture)
            st.image(cam_image, caption='Webcam Image.', use_column_width=True)

            if st.button("Predict Webcam Image"):
                predicted_label, confidence, top3 = _render_prediction(model, labels, cam_image)
                st.write(f"**Prediction:** {predicted_label} with {confidence*100:.2f}% confidence.")
                st.write("**Top-3:**")
                st.write({k: f"{v*100:.2f}%" for k, v in top3})

    # Real-time streaming mode (requires streamlit-webrtc)
    st.divider()
    st.subheader("Real-time Webcam (live video)")
    st.caption(
        "Chạy dự đoán liên tục trên từng frame. Tuỳ chọn crop theo hand: ưu tiên MediaPipe (nếu có), "
        "còn không thì dùng skin segmentation (OpenCV) để tăng độ ổn định."
    )

    try:
        import cv2  # type: ignore
    except Exception as e:
        st.info("Chưa có dependencies cho real-time webcam. Cài bằng: `pip install -r requirements.txt` rồi reload app.")
        st.caption(f"Chi tiết lỗi import: {e}")
        return

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase  # type: ignore
    except Exception as e:
        st.info("Chưa có `streamlit-webrtc`. Cài bằng: `pip install -r requirements.txt` rồi reload app.")
        st.caption(f"Chi tiết lỗi import: {e}")
        return

    mp = None
    hand_crop_supported = False
    try:
        import mediapipe as _mp  # type: ignore
        # Một số môi trường có thể import được mediapipe nhưng thiếu submodule cần thiết
        hand_crop_supported = hasattr(_mp, "solutions") and hasattr(getattr(_mp, "solutions", None), "hands")
        if hand_crop_supported:
            mp = _mp
        else:
            st.warning(
                "`mediapipe` hiện tại không có `solutions.hands` -> sẽ dùng skin segmentation (OpenCV) để crop hand trong Real-time."
            )
    except Exception as e:
        mp = None
        st.warning("Chưa load được `mediapipe` -> sẽ dùng skin segmentation (OpenCV) để crop hand trong Real-time.")
        st.caption(f"Chi tiết lỗi import: {e}")

    predict_every_n = st.sidebar.slider("Predict every N frames", min_value=1, max_value=30, value=5, step=1)
    show_top3 = st.sidebar.checkbox("Show Top-3 (overlay)", value=False)
    use_hand_crop = st.sidebar.checkbox("Crop hand ROI (hand detection)", value=True)
    hand_margin_ratio = st.sidebar.slider(
        "Hand crop margin",
        min_value=0.0,
        max_value=0.6,
        value=0.25,
        step=0.05,
        disabled=not use_hand_crop,
    )
    max_num_hands = st.sidebar.slider("Max hands", min_value=1, max_value=2, value=1, step=1, disabled=mp is None)
    skip_predict_if_no_hand = st.sidebar.checkbox(
        "Skip predict when no hand detected",
        value=True,
        disabled=not use_hand_crop,
    )
    allow_full_frame_predict = st.sidebar.checkbox(
        "Allow predict on full frame (when hand not detected)",
        value=False,
        disabled=(not use_hand_crop),
        help="Dataset của bạn chỉ học trên hand; để an toàn nên tắt mặc định.",
    )

    class VideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._frame_count = 0
            self._last_label = ""
            self._last_conf = 0.0
            self._last_top3: list[tuple[str, float]] = []
            self._last_bbox: tuple[int, int, int, int] | None = None
            self._hands = None
            self._skip_predict_if_no_hand = bool(skip_predict_if_no_hand)
            self._allow_full_frame_predict = bool(allow_full_frame_predict)
            self._use_hand_crop = bool(use_hand_crop)

            if self._use_hand_crop and mp is not None and hand_crop_supported:
                # Lightweight hand detector for realtime.
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=int(max_num_hands),
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self._frame_count += 1

            if self._frame_count % int(predict_every_n) == 0:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_for_predict = None
                bbox = None

                if self._use_hand_crop:
                    if self._hands is not None:
                        results = self._hands.process(rgb)
                        if results.multi_hand_landmarks:
                            bbox = _hand_bbox_square_from_landmarks(
                                results.multi_hand_landmarks[0],
                                img_w=rgb.shape[1],
                                img_h=rgb.shape[0],
                                margin_ratio=float(hand_margin_ratio),
                            )
                    else:
                        # Fallback: skin/hand ROI by OpenCV
                        bbox = _skin_bbox_square_from_bgr(
                            img_bgr=img,
                            margin_ratio=float(hand_margin_ratio),
                        )

                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        crop_rgb = rgb[y1:y2, x1:x2]
                        pil_for_predict = Image.fromarray(crop_rgb)

                if pil_for_predict is None:
                    # Dataset của bạn chỉ học trên HAND: nếu không crop được hand thì skip predict theo lựa chọn.
                    if self._skip_predict_if_no_hand and self._use_hand_crop and bbox is None:
                        self._last_label = ""
                        self._last_conf = 0.0
                        self._last_top3 = []
                        self._last_bbox = None
                    elif self._allow_full_frame_predict:
                        pil_for_predict = Image.fromarray(rgb)
                    else:
                        self._last_label = ""
                        self._last_conf = 0.0
                        self._last_top3 = []
                        self._last_bbox = None

                if pil_for_predict is not None:
                    label, conf, top3 = _render_prediction(model, labels, pil_for_predict)
                    self._last_label, self._last_conf, self._last_top3 = label, conf, top3
                    self._last_bbox = bbox

            # overlay
            text = f"{self._last_label} ({self._last_conf*100:.1f}%)" if self._last_label else "..."
            cv2.rectangle(img, (10, 10), (10 + 360, 10 + 30), (0, 0, 0), -1)
            cv2.putText(img, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if show_top3 and self._last_top3:
                y = 60
                for k, v in self._last_top3:
                    line = f"{k}: {v*100:.1f}%"
                    cv2.putText(img, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    y += 22

            if self._use_hand_crop and self._last_bbox is not None:
                x1, y1, x2, y2 = self._last_bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return frame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="asl-realtime",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
