from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS
import os
import tempfile
import torch
from transformers import pipeline
import time
import uuid
import inspect
from whisperspeech.pipeline import Pipeline
pipe = Pipeline()
from gtts import gTTS
from TTS.api import TTS
import numpy as np

import soundfile as sf  # For saving audio data

app = Flask(__name__, static_folder="static", template_folder="templates")

CORS(app)  # Enable CORS for all routes
       
# Initialize WhisperSpeech pipeline
print("Loading WhisperSpeech models...")
start_time = time.time()
pipe = Pipeline()  # Create the pipeline object
print(f"WhisperSpeech loaded in {time.time() - start_time:.2f} seconds")

print("Loading TTS model...")
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("TTS model loaded successfully")
except Exception as e:
    print(f"Error loading TTS model: {e}")
    tts = None
       
# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure supported models
MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3"
}

# Initialize with the smallest model - others will be loaded on demand
current_model_name = "tiny"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


print(f"Loaded model: {current_model_name}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/models')
def get_models():
    """Get available models"""
    global MODELS
    return jsonify({
        "models": list(MODELS.keys()),
        "current": current_model_name,
        "device": device
    })

@app.route('/api/change_model', methods=['POST'])
def change_model():
    """Change the current model"""
    global transcriber, current_model_name
    
    data = request.json
    model_name = data.get('model', 'tiny')
    
    if model_name not in MODELS:
        return jsonify({"error": f"Model {model_name} not found"}), 400
    
    if model_name != current_model_name:
        try:
            print(f"Loading model: {model_name}")
            start_time = time.time()
            
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load new model
            transcriber = pipeline(
                "automatic-speech-recognition", 
                model=MODELS[model_name],
                device=device,
                chunk_length_s=30,
                return_timestamps=True
            )
            
            current_model_name = model_name
            load_time = time.time() - start_time
            
            print(f"Model {model_name} loaded in {load_time:.2f} seconds")
            return jsonify({
                "success": True, 
                "model": model_name, 
                "load_time": f"{load_time:.2f} seconds"
            })
        
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    return jsonify({"success": True, "model": model_name, "message": "Model already loaded"})

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe uploaded audio file"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', None)
    
    # Generate a unique filename
    filename = f"{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the file
    audio_file.save(filepath)
    
    try:
        start_time = time.time()
        
        # Prepare transcription options
        # With Hugging Face pipeline, we need to use generate_kwargs for language
        if language:
            # For Whisper in Hugging Face, language is set through forced_decoder_ids
            # We'll recreate the pipeline with the right language setting
            transcribe_kwargs = {
                "generate_kwargs": {
                    "task": "transcribe",
                    "language": language
                }
            }
        else:
            transcribe_kwargs = {}
        
        # Transcribe the audio
        result = transcriber(filepath, **transcribe_kwargs)
        
        process_time = time.time() - start_time
        
        # Return results
        return jsonify({
            "success": True,
            "text": result["text"],
            "chunks": result.get("chunks", []),
            "language": language or "auto-detected",
            "model": current_model_name,
            "device": device,
            "process_time": f"{process_time:.2f} seconds",
            "file_url": f"/uploads/{filename}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

@app.route('/api/whisper/tts', methods=['POST'])
def text_to_speech_whisper():
    """Convert text to speech using WhisperSpeech"""
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    print(f"Debug - Text value: '{text}', Language: '{language}'")  # Debug print
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Generate speech
        start_time = time.time()
        
        # Use the correct parameter names as per the signature
        # fname, text, speaker=None, lang='en', cps=15, step_callback=None
        Pipeline().generate_to_file(
            fname=output_path,  # First parameter is fname
            text=text,          # Second parameter is text
            lang=language       # Language parameter
        )
        
        print(f"Speech generated in {time.time() - start_time:.2f} seconds")
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="speech.wav"
        )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Let the OS handle cleanup
        pass
                
@app.route('/api/google/tts', methods=['POST', 'OPTIONS'])
def text_to_speech_google():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    # Handle the actual POST request
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_filename = temp_audio.name
        
        # Generate speech using Google TTS
        tts = gTTS(text=text, lang=language)
        tts.save(temp_filename)
        
        # Send file
        return send_file(
            temp_filename,
            mimetype="audio/mp3",
            as_attachment=True,
            download_name="speech.mp3"
        )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file after sending
        if os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass

# Add a cleanup route to manually delete temporary files if needed
@app.route('/api/cleanup', methods=['POST'])
def cleanup_temp_files():
    """Clean up temporary files in the temp directory"""
    temp_dir = tempfile.gettempdir()
    count = 0
    
    try:
        for filename in os.listdir(temp_dir):
            if filename.endswith('.mp3') and os.path.isfile(os.path.join(temp_dir, filename)):
                try:
                    os.unlink(os.path.join(temp_dir, filename))
                    count += 1
                except:
                    pass
        
        return jsonify({"success": True, "message": f"Cleaned up {count} temporary files"})
    
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500


@app.route('/api/whisper/tts_clone', methods=['POST'])
def tts_clone_whisper():
    """Generate speech with voice cloning using WhisperSpeech"""
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    
    if 'voice_file' not in request.files:
        return jsonify({"error": "No voice reference file provided"}), 400
    
    print(f"Debug - Text value for cloning: '{text}', Language: '{language}'")  # Debug print
    
    # Save reference voice file
    voice_file = request.files['voice_file']
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_voice:
        voice_path = temp_voice.name
        voice_file.save(voice_path)
    
    try:
        # Create temporary file for output audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Generate speech with voice cloning
        start_time = time.time()
        
        # Use the correct parameter names as per the signature
        # fname, text, speaker=None, lang='en', cps=15, step_callback=None
        Pipeline().generate_to_file(
            fname=output_path,  # First parameter is fname
            text=text,          # Second parameter is text
            speaker=voice_path, # Use speaker parameter for voice cloning
            lang=language       # Language parameter
        )
        
        print(f"Speech generated in {time.time() - start_time:.2f} seconds")
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="cloned_speech.wav"
        )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except:
                pass
                
@app.route('/api/tts_clone', methods=['POST'])
def tts_clone():
    """Generate speech with voice cloning"""
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    if 'voice_file' not in request.files:
        return jsonify({"error": "No voice reference file provided"}), 400
    
    text = request.json.get('text')
    language = request.json.get('language', 'en')
    
    # Save reference voice file
    voice_file = request.files['voice_file']
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_voice:
        voice_path = temp_voice.name
        voice_file.save(voice_path)
    
    try:
        # Create temporary file for output audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Generate speech with voice cloning
        start_time = time.time()
        tts.tts_with_xtts_to_file(
            text=text,
            language=language,
            speaker_wav=voice_path,
            file_path=output_path
        )
        print(f"Speech generated in {time.time() - start_time:.2f} seconds")
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="cloned_speech.wav"
        )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except:
                pass

@app.route('/api/styletts/tts_clone', methods=['POST'])
def tts_styletts2():
    """Generate speech with StyleTTS2"""
    if tts_model is None:
        return jsonify({"error": "StyleTTS2 model not available"}), 500
        
    if 'text' not in request.form:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.form.get('text')
    
    try:
        # Create temporary file for output audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Generate speech
        start_time = time.time()
        
        # Try different possible ways to generate speech
        if hasattr(tts_model, "synthesize"):
            audio = tts_model.synthesize(text)
            # Save audio to file
            with open(output_path, 'wb') as f:
                f.write(audio)
        elif hasattr(tts_model, "tts"):
            tts_model.tts(text, output_path=output_path)
        elif hasattr(tts_model, "inference"):
            tts_model.inference(text, output_path=output_path)
        else:
            return jsonify({"error": "Could not find synthesis method in StyleTTS2"}), 500
        
        print(f"Speech generated in {time.time() - start_time:.2f} seconds")
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="speech.wav"
        )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Clean up will be handled by the OS
        pass

@app.route('/api/coqui/tts', methods=['POST'])
def text_to_speech_coqui():
    """Convert text to speech using TTS"""
    if tts is None:
        return jsonify({"error": "TTS model not available"}), 500
        
    if 'text' not in request.form:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.form.get('text')
    language = request.form.get('language', 'en')
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Generate speech
        start_time = time.time()
        
        # Use tts_to_file instead of assuming write_to_file exists
        tts.tts_to_file(text=text, file_path=output_path, language=language)
        
        print(f"Speech generated in {time.time() - start_time:.2f} seconds")
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="speech.wav"
        )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Let the OS handle cleanup
        pass
        
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
