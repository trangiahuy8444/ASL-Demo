# ASL Streamlit Demo (Inference)

Demo nhận diện ký hiệu **American Sign Language (ASL)** bằng Streamlit, dùng model bạn đã train từ notebook `asl-demo.ipynb`.

## Pipeline tổng quan

- **Train (Notebook)**: `asl-demo.ipynb`
  - Load dataset từ thư mục `dataset/` (train/val split).
  - Model: Transfer Learning **MobileNetV2**.
  - Ảnh input: resize **224×224**, chuẩn hoá **/255**.
  - Lưu artifacts:
    - `best_asl_model.h5`: checkpoint tốt nhất theo `val_accuracy`
    - `asl_model_final.h5`: model cuối cùng sau training
    - `class_names.pkl`: danh sách label theo thứ tự output của model

- **Demo (Streamlit)**: `app.py`
  - Chọn model (sidebar) và dự đoán từ:
    - Upload ảnh (`jpg/jpeg/png`)
    - Webcam (chụp 1 frame)
    - Real-time webcam (live video, tùy chọn crop hand bằng MediaPipe)
  - Preprocess khớp với lúc train: resize 224×224, float32, chia 255.
  - Label được load từ `class_names.pkl` (fallback sang danh sách mặc định nếu thiếu file).

## Cấu trúc file chính

- `app.py`: Streamlit app để demo inference
- `asl-demo.ipynb`: notebook train/evaluate + lưu model
- `best_asl_model.h5`, `asl_model_final.h5`, `epoch_*.h5`: model đã train
- `class_names.pkl`: mapping index → label
- `dataset/`: dữ liệu train/test (nếu bạn giữ trong repo local)
- `requirements.txt`: dependencies tối thiểu để chạy demo

## Cách chạy demo

1) Tạo môi trường và cài dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Chạy Streamlit:

```bash
streamlit run app.py
```

3) Trong app:

- Ở sidebar, chọn model muốn dùng (thường là `best_asl_model.h5`).
- Chọn **Upload Image** hoặc **Use Webcam** rồi bấm **Predict**.

## Ghi chú

- Vì model được train trên ảnh chỉ có `hand` (không có object), trong chế độ **Real-time Webcam** khi bật `Crop hand`:
  - Nếu không detect được tay thì app sẽ **không predict** (mặc định bật).
- Nếu `mediapipe` trong môi trường của bạn không hỗ trợ `solutions.hands`, app sẽ **dùng skin segmentation (OpenCV)** để crop hand (real-time vẫn chạy và predict theo ROI).
- Nếu model của bạn được train với kích thước input khác 224×224, hãy chỉnh `input_size` trong `preprocess_pil()` của `app.py` cho khớp.
- Nếu bạn train lại và label order thay đổi, hãy nhớ cập nhật `class_names.pkl` tương ứng (app sẽ ưu tiên file này).

