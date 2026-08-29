# app.py
from flask import Flask, request, send_file, jsonify, after_this_request
from werkzeug.utils import secure_filename
import os
from another import Audio8DConverter, AudioConfig
import logging
from datetime import datetime
from pathlib import Path
import threading

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
CLEANUP_DELAY_SECONDS = 60  # Delay before cleaning up processed files

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Configure maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def schedule_cleanup(filepath, delay=CLEANUP_DELAY_SECONDS):
    """Schedule a file for cleanup after a delay."""
    def cleanup():
        try:
            import time
            time.sleep(delay)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to cleanup file {filepath}: {e}")
    
    thread = threading.Thread(target=cleanup, daemon=True)
    thread.start()

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
    """Handle audio conversion request."""
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
        
        input_path = Path(app.config['UPLOAD_FOLDER']) / f"{base_filename}_{timestamp}.mp3"
        output_path = Path(app.config['PROCESSED_FOLDER']) / f"{base_filename}_{timestamp}_8d.mp3"

        # Save the uploaded file
        file.save(str(input_path))
        logger.info(f"File saved: {input_path}")

        # Get parameters from form with validation
        try:
            pan_speed = float(request.form.get('pan_speed', 0.5))
            depth = float(request.form.get('depth', 0.95))
            reverb_delay = int(request.form.get('reverb_delay', 50))
            reverb_decay = float(request.form.get('reverb_decay', 0.3))
            
            config = AudioConfig(
                pan_speed=pan_speed,
                depth=depth,
                reverb_delay=reverb_delay,
                reverb_decay=reverb_decay
            )
        except ValueError as e:
            return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400

        # Convert the file
        converter = Audio8DConverter(config)
        converter.convert_file(str(input_path), str(output_path))
        
        # Schedule cleanup of input file immediately
        schedule_cleanup(str(input_path), delay=0)
        
        # Schedule cleanup of output file after download delay
        schedule_cleanup(str(output_path), delay=CLEANUP_DELAY_SECONDS)

        # Send the processed file with automatic cleanup after sending
        @after_this_request
        def remove_output(response):
            """Remove output file after response is sent (as backup to scheduled cleanup)."""
            try:
                if output_path.exists():
                    os.remove(str(output_path))
                    logger.info(f"Removed output file after download: {output_path}")
            except Exception as e:
                logger.error(f"Error removing output file: {e}")
            return response
        
        return send_file(
            str(output_path),
            as_attachment=True,
            download_name=f"{base_filename}_8d.mp3",
            mimetype='audio/mpeg'
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return jsonify({'error': str(e)}), 404
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in conversion: {str(e)}")
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors."""
    return jsonify({'error': 'Resource not found'}), 404

if __name__ == '__main__':
    # Use environment variable for port in production
    port = int(os.environ.get('PORT', 5000))
    # Debug mode should be disabled in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)