from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from scanner.document_scanner import scan_document, get_document_info
import cv2
import numpy as np
import io
import os
from PIL import Image
import uuid
from werkzeug.utils import secure_filename
import time
from datetime import datetime
import magic
import hashlib
import re

app = Flask(__name__)

# Security configurations
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_MIMETYPES'] = {'image/jpeg', 'image/png', 'image/gif', 'image/bmp'}

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False, 
                'error': 'No image uploaded'
            }), 400
        
        file = request.files['image']
        
        # Handle camera capture (usually larger files)
        is_camera = request.form.get('source') == 'camera'
        if is_camera:
            app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB for camera
        
        # Validate file
        if not file or not file.filename:
            return jsonify({
                'success': False, 
                'error': 'Invalid file'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': 'File type not allowed'
            }), 400
        
        if not validate_file_contents(file):
            return jsonify({
                'success': False, 
                'error': 'Invalid file contents'
            }), 400
        
        # Read the image safely
        try:
            image_bytes = file.read()
            if len(image_bytes) > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({
                    'success': False, 
                    'error': 'File too large'
                }), 400
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': 'Error reading file'
            }), 400
        
        # Get document info first
        doc_info = get_document_info(image_bytes)
        
        if not doc_info['success']:
            return jsonify({
                'success': False,
                'error': 'Could not detect document in image'
            }), 400
        
        # Get enhancement preferences
        enhance = request.form.get('enhance', 'true').lower() == 'true'
        auto_rotate = request.form.get('auto_rotate', 'true').lower() == 'true'
        
        # Process the image with enhancements
        scanned = scan_document(image_bytes, enhance=enhance)
        
        if scanned is None:
            return jsonify({
                'success': False,
                'error': 'Could not process document'
            }), 400
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_prefix = 'cam' if is_camera else 'upload'
        filename = f"{source_prefix}_{timestamp}.png"
        filename = sanitize_filename(filename)
        
        # Save the processed image
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            cv2.imwrite(temp_path, scanned)
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': 'Error saving file'
            }), 500

        return jsonify({
            'success': True,
            'image_url': f'/uploads/{filename}',
            'doc_info': {
                'confidence': doc_info['confidence'],
                'angle': doc_info['angle']
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scan', methods=['POST'])
def api_scan():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Read the image
        image_bytes = file.read()
        
        # Process the image
        scanned = scan_document(image_bytes)
        
        if scanned is None:
            return jsonify({'error': 'Could not detect document in image'}), 400
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', scanned)
        image_base64 = buffer.tobytes()
        
        return send_file(
            io.BytesIO(image_base64),
            mimetype='image/png',
            as_attachment=True,
            download_name='scanned_document.png'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; \
        script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; \
        style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; \
        img-src 'self' data: blob:; \
        font-src cdnjs.cloudflare.com"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

def allowed_file(filename):
    """Check if file extension and mimetype are allowed"""
    if not '.' in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    if ext not in app.config['UPLOAD_EXTENSIONS']:
        return False
    return True

def validate_file_contents(file_stream):
    """Validate file contents using basic image validation"""
    try:
        # Read the first few bytes to check file signature
        header = file_stream.read(8)
        file_stream.seek(0)  # Reset file pointer
        
        # Check for common image signatures
        png_signature = b'\x89PNG\r\n\x1a\n'
        jpeg_signatures = [b'\xFF\xD8\xFF', b'\xFF\xD8\xFF\xE0', b'\xFF\xD8\xFF\xE1']
        gif_signatures = [b'GIF87a', b'GIF89a']
        bmp_signature = b'BM'
        
        # Check PNG
        if header.startswith(png_signature):
            return True
            
        # Check JPEG
        for sig in jpeg_signatures:
            if header.startswith(sig):
                return True
                
        # Check GIF
        for sig in gif_signatures:
            if header.startswith(sig):
                return True
                
        # Check BMP
        if header.startswith(bmp_signature):
            return True
            
        return False
        
    except Exception:
        return False

def sanitize_filename(filename):
    """Sanitize filename to prevent path traversal"""
    filename = secure_filename(filename)
    # Additional sanitization
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    # Add random hash to prevent filename collisions
    name, ext = os.path.splitext(filename)
    hash_str = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"{name}_{hash_str}{ext}"

# Add route to serve temporary files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Validate filename
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        return 'Invalid filename', 400
    
    # Prevent directory traversal
    filename = os.path.basename(filename)
    
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename, 
            as_attachment=True,
            mimetype='image/png'
        )
    except Exception:
        return 'File not found', 404

def cleanup_old_files():
    """Remove files older than 1 hour"""
    try:
        now = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
                continue  # Skip files with suspicious names
                
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.getmtime(filepath) < now - 3600:  # 1 hour
                    if os.path.isfile(filepath):  # Extra check
                        os.remove(filepath)
            except:
                continue
    except:
        pass  # Fail silently

# Add to the scan route
@app.before_request
def before_request():
    cleanup_old_files()

@app.route('/gallery')
def gallery():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            timestamp = datetime.fromtimestamp(os.path.getmtime(filepath))
            files.append({
                'name': filename,
                'url': f'/uploads/{filename}',
                'date': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
    # Sort files by date, newest first
    files.sort(key=lambda x: x['date'], reverse=True)
    return render_template('gallery.html', files=files)

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    try:
        # Validate filename
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return jsonify({'success': False, 'error': 'Invalid filename'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete-all', methods=['POST'])
def delete_all_files():
    try:
        deleted = 0
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    deleted += 1
                except:
                    continue
        return jsonify({'success': True, 'count': deleted})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/guide')
def guide():
    return render_template('guide.html')

if __name__ == '__main__':
    app.run(debug=True) 