from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from scanner.document_scanner import scan_document, get_document_info, allowed_file, MAX_IMAGE_SIZE
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
import logging
import json
import shutil
import sys

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
    temp_id = None
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Check if this is a preview update request
        is_preview_update = request.form.get('is_preview_update') == 'true'
        if is_preview_update:
            temp_id = request.form.get('temp_id')
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', f'original_{temp_id}.png')
            if not os.path.exists(original_path):
                return jsonify({'success': False, 'error': 'Original file not found'}), 404
                
            # Process using original file for preview
            try:
                options = json.loads(request.form.get('options', '{}'))
                scan_options = {
                    'mode': options.get('mode', 'original'),
                    'contrast': float(options.get('contrast', 1.2)),
                    'clean_background': bool(options.get('clean_background', True)),
                    'straighten': bool(options.get('straighten', True)),
                    'sharpen_text': bool(options.get('sharpen_text', True))
                }
                
                # Process the image using original
                result = scan_document(original_path, enhance=True, options=scan_options)
                if not result['success']:
                    raise Exception(result.get('error', 'Failed to process document'))
                    
                # Save new preview
                preview_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', f'processed_{temp_id}.png')
                if not cv2.imwrite(preview_path, result['processed']):
                    raise Exception("Failed to save preview")
                    
                return jsonify({
                    'success': True,
                    'temp_id': temp_id,
                    'preview_url': f'/uploads/temp/processed_{temp_id}.png?t={int(time.time())}',
                    'is_temporary': True
                })
                
            except Exception as e:
                error_msg = str(e) if str(e) != '' else 'Failed to update preview'
                return jsonify({'success': False, 'error': error_msg}), 400

        # Handle new document upload
        if not file or not file.filename:
            return jsonify({'success': False, 'error': 'Invalid file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400

        # Read file safely
        try:
            image_bytes = file.read()
            if not image_bytes:
                return jsonify({'success': False, 'error': 'Empty file'}), 400
                
            # Validate image data
            nparr = np.frombuffer(image_bytes, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if original is None:
                return jsonify({'success': False, 'error': 'Invalid image format'}), 400
        except Exception as e:
            logging.error(f"File reading failed: {str(e)}")
            return jsonify({'success': False, 'error': 'Could not read image file'}), 400

        # Generate temp ID and setup directory
        temp_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save original image
            original_filename = f"original_{temp_id}.png"
            original_path = os.path.join(temp_dir, original_filename)
            
            if not cv2.imwrite(original_path, original):
                raise Exception("Failed to save uploaded image")

            # Get enhancement options
            options = json.loads(request.form.get('options', '{}'))
            scan_options = {
                'mode': options.get('mode', 'original'),
                'contrast': float(options.get('contrast', 1.2)),
                'clean_background': bool(options.get('clean_background', True)),
                'straighten': bool(options.get('straighten', True)),
                'sharpen_text': bool(options.get('sharpen_text', True))
            }

            # Process the image
            result = scan_document(original_path, enhance=True, options=scan_options)
            
            if not result['success']:
                raise Exception(result.get('error', 'Failed to process document'))

            # Save processed version
            processed_filename = f"processed_{temp_id}.png"
            processed_path = os.path.join(temp_dir, processed_filename)
            
            if not cv2.imwrite(processed_path, result['processed']):
                raise Exception("Failed to save processed image")

            return jsonify({
                'success': True,
                'temp_id': temp_id,
                'preview_url': f'/uploads/temp/processed_{temp_id}.png?t={int(time.time())}',
                'original_url': f'/uploads/temp/original_{temp_id}.png',
                'is_temporary': True
            })

        except Exception as e:
            if temp_id:
                cleanup_temp_session(temp_id)
            error_msg = str(e) if str(e) != '' else 'Failed to process document'
            return jsonify({'success': False, 'error': error_msg}), 400

    except Exception as e:
        if temp_id:
            cleanup_temp_session(temp_id)
        logging.error(f"Scan failed: {str(e)}")
        return jsonify({'success': False, 'error': 'An error occurred while processing the document'}), 500

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

@app.route('/uploads/temp/<filename>')
def serve_temp_file(filename):
    """Serve temporary files from the temp directory"""
    try:
        # Validate filename
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return 'Invalid filename', 400
        
        # Prevent directory traversal
        filename = os.path.basename(filename)
        
        # Serve from temp directory
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        return send_from_directory(temp_dir, filename, as_attachment=True)
    except Exception:
        return 'File not found', 404

def cleanup_temp_files():
    """Remove temporary files older than 1 hour"""
    try:
        now = time.time()
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
                    continue  # Skip files with suspicious names
                
                filepath = os.path.join(temp_dir, filename)
            try:
                if os.path.getmtime(filepath) < now - 3600:  # 1 hour
                        if os.path.isfile(filepath):
                            os.remove(filepath)
            except:
                pass  # Fail silently
    except:
        pass  # Fail silently

# Add to the before_request handler
@app.before_request
def before_request():
    cleanup_temp_files()

def generate_default_filename(index=None):
    """Generate a default filename for documents"""
    if index is None:
        return f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"image{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

@app.route('/gallery')
def gallery():
    try:
        files = []
        upload_dir = app.config['UPLOAD_FOLDER']
        
        if os.path.exists(upload_dir):
            # Get all files and their info
            for filename in os.listdir(upload_dir):
                if os.path.isfile(os.path.join(upload_dir, filename)) and not filename.startswith('.'):
                    file_path = os.path.join(upload_dir, filename)
                    stat = os.stat(file_path)
                    
                    # Get file creation/modification time
                    timestamp = stat.st_mtime
                    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Get display name (remove timestamp from filename)
                    display_name = filename.split('_')[0] if '_' in filename else filename
                    if display_name.startswith('image'):
                        # For default names, keep the full name with timestamp
                        display_name = filename.rsplit('.', 1)[0]
                    
                    files.append({
                        'name': filename,
                        'display_name': display_name,
                        'url': f'/uploads/{filename}',
                        'date': date_str
                    })
            
            # Sort files by date (newest first)
            files.sort(key=lambda x: x['date'], reverse=True)
        
        return render_template('gallery.html', files=files)
        
    except Exception as e:
        logging.error(f"Gallery load failed: {str(e)}")
        return render_template('gallery.html', files=[])

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

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        if request.content_length > MAX_IMAGE_SIZE:
            return jsonify({'error': 'File size too large'}), 400
            
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        try:
            file.save(temp_path)
            result = scan_document(temp_path)
            return jsonify(result)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logging.error(f"Upload failed: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/edit/<filename>')
def edit_document(filename):
    """Edit page for saved documents"""
    try:
        # Validate filename
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return 'Invalid filename', 400
            
        # Check if file exists
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return 'File not found', 404
            
        # Generate temp ID for editing session
        temp_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy file to temp directory for editing
        temp_original = f"original_{temp_id}.png"
        temp_path = os.path.join(temp_dir, temp_original)
        shutil.copy2(filepath, temp_path)
        
        return render_template('edit.html', 
                             filename=filename,
                             temp_id=temp_id,
                             original_url=f'/uploads/temp/{temp_original}')
                             
    except Exception as e:
        logging.error(f"Edit page failed: {str(e)}")
        return 'Error loading edit page', 500

@app.route('/save_edited', methods=['POST'])
def save_edited():
    """Save edited document back to gallery"""
    try:
        data = request.json
        temp_id = data.get('temp_id')
        original_filename = data.get('original_filename')
        
        if not temp_id or not original_filename:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400
            
        # Get paths
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', f'processed_{temp_id}.png')
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        
        # Replace original with edited version
        if os.path.exists(processed_path):
            shutil.move(processed_path, final_path)
            
            # Clean up all temp files for this session
            cleanup_temp_session(temp_id)
            
            return jsonify({
                'success': True,
                'message': 'Changes saved successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Edited file not found'}), 404
            
    except Exception as e:
        logging.error(f"Save edited failed: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to save changes'}), 500

def cleanup_temp_session(temp_id):
    """Remove all temporary files for a specific session"""
    try:
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        if not os.path.exists(temp_dir):
            return
            
        # First, remove specific temp files for this session
        for filename in os.listdir(temp_dir):
            if temp_id in filename:
                filepath = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                        logging.info(f"Cleaned up temp file: {filename}")
                except Exception as e:
                    logging.error(f"Failed to remove temp file {filename}: {str(e)}")
        
        # Check if directory is empty and remove it
        try:
            remaining_files = os.listdir(temp_dir)
            if not remaining_files:
                os.rmdir(temp_dir)
                logging.info("Removed empty temp directory")
        except Exception as e:
            logging.error(f"Failed to check/remove temp directory: {str(e)}")
            
    except Exception as e:
        logging.error(f"Cleanup failed: {str(e)}")

@app.route('/save_document', methods=['POST'])
def save_document():
    try:
        data = request.json
        temp_id = data.get('temp_id')
        custom_name = data.get('name', '').strip()
        
        if not temp_id:
            return jsonify({'success': False, 'error': 'No temporary ID provided'}), 400
            
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if custom_name:
            # Use custom name if provided
            safe_name = secure_filename(custom_name)
            final_filename = f"{safe_name}_{timestamp}.png"
        else:
            # Generate default name
            existing_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                            if f.startswith('image') and f.endswith('.png')]
            index = len(existing_files) + 1
            final_filename = f"{generate_default_filename(index)}.png"
            
        final_filename = sanitize_filename(final_filename)
        
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        processed_path = os.path.join(temp_dir, f"processed_{temp_id}.png")
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        
        if not os.path.exists(processed_path):
            cleanup_temp_session(temp_id)
            return jsonify({'success': False, 'error': 'Temporary file not found'}), 404
            
        try:
            # Copy processed file to gallery
            shutil.copy2(processed_path, final_path)
            
            # Verify the copy was successful
            if not os.path.exists(final_path):
                raise Exception("Failed to save to gallery")
                
            # Clean up temp files
            cleanup_temp_session(temp_id)
            
            return jsonify({
                'success': True,
                'image_url': f'/uploads/{final_filename}',
                'is_temporary': False
            })
            
        except Exception as e:
            logging.error(f"Failed to save document: {str(e)}")
            cleanup_temp_session(temp_id)
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                except:
                    pass
            return jsonify({'success': False, 'error': 'Failed to save document'}), 500
            
    except Exception as e:
        logging.error(f"Save failed: {str(e)}")
        if temp_id:
            cleanup_temp_session(temp_id)
        return jsonify({'success': False, 'error': 'Failed to save document'}), 500

@app.route('/cleanup_temp', methods=['POST'])
def cleanup_temp():
    """Clean up temporary files for a session"""
    try:
        data = request.json
        temp_id = data.get('temp_id')
        
        if temp_id:
            cleanup_temp_session(temp_id)
            
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Cleanup request failed: {str(e)}")
        return jsonify({'success': False}), 500

@app.route('/preview_update', methods=['POST'])
def preview_update():
    """Handle real-time preview updates"""
    try:
        data = request.json
        temp_id = data.get('temp_id')
        options = data.get('options', {})
        
        if not temp_id:
            return jsonify({'success': False, 'error': 'Missing temp_id'}), 400
            
        # Get paths
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        original_path = os.path.join(temp_dir, f'original_{temp_id}.png')
        
        if not os.path.exists(original_path):
            return jsonify({'success': False, 'error': 'Original file not found'}), 404
            
        try:
            # Process the image with new options
            scan_options = {
                'mode': options.get('mode', 'original'),
                'contrast': float(options.get('contrast', 1.2)),
                'clean_background': bool(options.get('clean_background', True)),
                'straighten': bool(options.get('straighten', True)),
                'sharpen_text': bool(options.get('sharpen_text', True)),
                'auto_crop': True  # Always enable auto-crop for best results
            }
            
            result = scan_document(original_path, enhance=True, options=scan_options)
            
            if not result['success']:
                raise Exception(result.get('error', 'Failed to process document'))
                
            # Save new preview
            preview_path = os.path.join(temp_dir, f'processed_{temp_id}.png')
            if not cv2.imwrite(preview_path, result['processed']):
                raise Exception("Failed to save preview")
                
            timestamp = int(time.time() * 1000)
            return jsonify({
                'success': True,
                'preview_url': f'/uploads/temp/processed_{temp_id}.png?t={timestamp}',
                'is_temporary': True
            })
            
        except Exception as e:
            error_msg = str(e) if str(e) != '' else 'Failed to update preview'
            return jsonify({'success': False, 'error': error_msg}), 400
            
    except Exception as e:
        logging.error(f"Preview update failed: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update preview'}), 500

if __name__ == '__main__':
    app.run(debug=True) 