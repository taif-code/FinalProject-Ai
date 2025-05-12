from flask import Flask, render_template, request, jsonify
import os
from your_ai_code import audio_tool_func, image_tool_func, rag_tool_func  # افصل الأكواد هنا لو أحببت

app = Flask(__name__)
print("✅ Flask has started and app is configured.")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    query = request.form.get('text')
    if not query:
        return jsonify({'error': 'No text provided'}), 400
    answer, lang, source = rag_tool_func(query)
    return jsonify({'answer': answer, 'source': source})

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    audio = request.files.get('audio')
    if not audio:
        return jsonify({'error': 'No audio uploaded'}), 400
    path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(path)
    answer, lang, source = audio_tool_func(path)
    return jsonify({'answer': answer, 'source': source})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    image = request.files.get('image')
    if not image:
        return jsonify({'error': 'No image uploaded'}), 400
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)
    answer, lang, source = image_tool_func(path)
    return jsonify({'answer': answer, 'source': source})

if __name__ == '__main__':
    app.run(debug=True)
    
