# 🚀 GUI Usage (Streamlit Interface)
This project implements the SIGGRAPH17 Deep Learning Colorization model (based on Zang's release) with a simplified and user-friendly GUI built using Streamlit.
The application displays three outputs:
1. Original Image
2. SIGGRAPH17 Colorized Result
3. Enhanced Result (CLAHE + RGB Boost)

# 📦 Instalasi
 Install the necessary dependencies:
 
```bash
pip install streamlit torch torchvision pillow opencv-python numpy
```
Or install from the requirements.txt file:
```bash
pip install -r requirements.txt
```
# Dependencies:
- Streamlit — web-based GUI
- PyTorch — deep learning inference
- OpenCV — enhancement + color conversion
- Pillow — image handling
- NumPy — array operations

# Ensure folder structure:
```bash
colorizers/
    siggraph17.py
    util.py
enhance.py
gui/
    streamlit.py
```
# ▶ 2. Run the Streamlit App
From the project directory
```bash
streamlit run gui/streamlit.py
```
Streamlit will open:
```bash
http://localhost:8501
```
# 🖼 3. How to Use
1. Upload an image
2. Click “Process Color (SIGGRAPH17)”
3. Click “Enhance”
4. Compare all 3 results (Original, SIGGRAPH, Enhanced)
