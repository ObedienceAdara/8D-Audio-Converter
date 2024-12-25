# app.py
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from another import Audio8DConverter, AudioConfig
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp3'}

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Configure maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>8D Audio Converter</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px;
                }
                .container { 
                    background: #f5f5f5; 
                    padding: 20px; 
                    border-radius: 8px;
                }
                form { margin: 20px 0; }
                .parameters {
                    margin: 15px 0;
                    padding: 10px;
                    background: #fff;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>8D Audio Converter</h1>
                <p>Upload an MP3 file to convert it to 8D audio.</p>
                <form action="/convert" method="post" enctype="multipart/form-data">
                    <div>
                        <input type="file" name="file" accept=".mp3" required>
                    </div>
                    <div class="parameters">
                        <h3>Parameters (optional):</h3>
                        <div>
                            <label>Pan Speed (0-2 Hz):
                                <input type="number" name="pan_speed" step="0.1" min="0.1" max="2" value="0.5">
                            </label>
                        </div>
                        <div>
                            <label>Depth (0-1):
                                <input type="number" name="depth" step="0.05" min="0" max="1" value="0.95">
                            </label>
                        </div>
                        <div>
                            <label>Reverb Delay (0-100ms):
                                <input type="number" name="reverb_delay" min="1" max="100" value="50">
                            </label>
                        </div>
                        <div>
                            <label>Reverb Decay (0-1):
                                <input type="number" name="reverb_decay" step="0.05" min="0" max="1" value="0.3">
                            </label>
                        </div>
                    </div>
                    <div>
                        <input type="submit" value="Convert to 8D">
                    </div>
                </form>
            </div>
        </body>
    </html>
    '''

@app.route('/convert', methods=['POST'])
def convert_audio():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only MP3 files are allowed'}), 400

        # Secure the filename and generate paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        base_filename = os.path.splitext(filename)[0]
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_{timestamp}.mp3")
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base_filename}_{timestamp}_8d.mp3")

        # Save the uploaded file
        file.save(input_path)
        logger.info(f"File saved: {input_path}")

        # Get parameters from form
        config = AudioConfig(
            pan_speed=float(request.form.get('pan_speed', 0.5)),
            depth=float(request.form.get('depth', 0.95)),
            reverb_delay=int(request.form.get('reverb_delay', 50)),
            reverb_decay=float(request.form.get('reverb_decay', 0.3))
        )

        # Convert the file
        converter = Audio8DConverter(config)
        converter.convert_file(input_path, output_path)
        
        # Clean up the input file
        os.remove(input_path)
        
        # Send the processed file
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"{base_filename}_8d.mp3",
            mimetype='audio/mpeg'
        )

    except Exception as e:
        logger.error(f"Error in conversion: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)