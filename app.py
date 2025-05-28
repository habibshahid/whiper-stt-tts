from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS
import os
import tempfile
import torch
from transformers import pipeline
import time
import uuid
import sys
import logging
import traceback 

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load WhisperSpeech pipeline
try:
    from whisperspeech.pipeline import Pipeline
    pipe = Pipeline()
    logger.info("WhisperSpeech loaded successfully")
    whisper_available = True
except Exception as e:
    logger.error(f"Error loading WhisperSpeech: {e}")
    whisper_available = False
    pipe = None

# Try loading gTTS
try:
    from gtts import gTTS
    gtts_available = True
    logger.info("gTTS loaded successfully")
except Exception as e:
    logger.error(f"Error loading gTTS: {e}")
    gtts_available = False

# IMPORTANT: Skip TTS import completely
# The bangla library has a Python 3.10+ syntax that fails in Python 3.9
# We'll just disable this functionality entirely
# Try loading TTS with verbose error handling
# Try loading TTS with verbose error handling
# Try loading TTS with verbose error handling
# Replace this section in your app.py

# Try loading TTS with verbose error handling - specifically targeting XTTS
tts = None
tts_available = False
tts_is_multilingual = False
try:
    logger.info("Attempting to load TTS library...")
    from TTS.api import TTS
    logger.info("TTS module imported successfully")
    
    # Set the XTTS model name
    xtts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    try:
        # First, try to load the XTTS model directly
        logger.info(f"Attempting to load XTTS model: {xtts_model_name}")
        tts = TTS(xtts_model_name)
        
        # Check model properties
        model_name = getattr(tts, 'model_name', 'unknown')
        is_multi_lingual = getattr(tts, 'is_multi_lingual', False)
        is_multi_speaker = getattr(tts, 'is_multi_speaker', False)
        
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"is_multi_lingual: {is_multi_lingual}")
        logger.info(f"is_multi_speaker: {is_multi_speaker}")
        
        # Check available methods for voice cloning
        has_xtts_method = hasattr(tts, 'tts_with_xtts_to_file')
        logger.info(f"Has tts_with_xtts_to_file method: {has_xtts_method}")
        
        # Set flags based on model properties
        tts_available = True
        tts_is_multilingual = is_multi_lingual
        
        # If model doesn't support voice cloning, log a warning
        if not is_multi_speaker:
            logger.warning("The loaded model does not support voice cloning (speaker conditioning)")
            
        logger.info("XTTS model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading XTTS model: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try loading a YourTTS model (which supports voice cloning) if XTTS fails
        try:
            logger.info("Trying to load YourTTS model as fallback")
            your_tts_model = "tts_models/multilingual/multi-dataset/your_tts"
            tts = TTS(your_tts_model)
            
            # Check model properties
            model_name = getattr(tts, 'model_name', 'unknown')
            is_multi_lingual = getattr(tts, 'is_multi_lingual', False)
            is_multi_speaker = getattr(tts, 'is_multi_speaker', False)
            
            logger.info(f"Loaded YourTTS model: {model_name}")
            logger.info(f"is_multi_lingual: {is_multi_lingual}")
            logger.info(f"is_multi_speaker: {is_multi_speaker}")
            
            # Set flags based on model properties
            tts_available = True
            tts_is_multilingual = is_multi_lingual
            
            logger.info("YourTTS model loaded successfully")
            
        except Exception as e2:
            logger.error(f"Error loading YourTTS model: {str(e2)}")
            
            # Try loading any tacotron model that supports speaker conditioning
            try:
                logger.info("Trying to load a fallback model with voice cloning support")
                tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
                
                # Check model properties
                model_name = getattr(tts, 'model_name', 'unknown')
                is_multi_lingual = getattr(tts, 'is_multi_lingual', False)
                is_multi_speaker = getattr(tts, 'is_multi_speaker', False)
                
                logger.info(f"Loaded fallback model: {model_name}")
                logger.info(f"is_multi_lingual: {is_multi_lingual}")
                logger.info(f"is_multi_speaker: {is_multi_speaker}")
                
                # Set flags based on model properties
                tts_available = True
                tts_is_multilingual = is_multi_lingual
                
                logger.info("Fallback model loaded successfully")
                
            except Exception as e3:
                logger.error(f"Error loading any TTS model: {str(e3)}")
                logger.warning("Voice cloning will not be available")
                
    except Exception as e:
        logger.error(f"Unexpected error initializing TTS: {str(e)}")
        logger.error(traceback.format_exc())
except ImportError as ie:
    logger.error(f"ImportError loading TTS module: {str(ie)}")
    logger.error(traceback.format_exc())
except Exception as outer_e:
    logger.error(f"Outer exception in TTS loading block: {str(outer_e)}")
    logger.error(traceback.format_exc())

# Log summary of TTS initialization
if tts_available:
    logger.info(f"TTS initialized successfully - multilingual: {tts_is_multilingual}")
else:
    logger.warning("TTS initialization failed - voice cloning will not be available")
    

# Create the Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes

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
logger.info(f"Using device: {device}")

# Initialize the transcriber
transcriber = pipeline(
    "automatic-speech-recognition", 
    model=MODELS[current_model_name],
    device=device,
    chunk_length_s=30,
    return_timestamps=True
)
logger.info(f"Loaded model: {current_model_name}")

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
        "device": device,
        "tts_available": tts_available,
        "whisper_available": whisper_available,
        "gtts_available": gtts_available
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
            logger.info(f"Loading model: {model_name}")
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
            
            logger.info(f"Model {model_name} loaded in {load_time:.2f} seconds")
            return jsonify({
                "success": True, 
                "model": model_name, 
                "load_time": f"{load_time:.2f} seconds"
            })
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
        logger.error(f"Transcription failed: {e}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

@app.route('/api/xtts/tts', methods=['POST', 'OPTIONS'])
def text_to_speech_xtts():
    """Convert text to speech using XTTS without voice cloning"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    if not tts_available:
        return jsonify({
            "error": "TTS is not available on this server",
            "suggestion": "Check server logs for details on TTS initialization"
        }), 503
    
    # Handle the actual POST request
    if not request.is_json:
        return jsonify({"error": "Expected JSON request"}), 400
        
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    logger.info(f"XTTS TTS request - Text: '{text[:50]}...', Language: '{language}'")
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Get information about the model
        is_multi_speaker = getattr(tts, 'is_multi_speaker', False)
        is_multi_lingual = getattr(tts, 'is_multi_lingual', False)
        model_name = getattr(tts, 'model_name', 'unknown')
        
        logger.info(f"Using model: {model_name}")
        logger.info(f"is_multi_speaker: {is_multi_speaker}")
        logger.info(f"is_multi_lingual: {is_multi_lingual}")
        
        # Determine which synthesis method to use
        has_tts = hasattr(tts, 'tts')
        has_tts_to_file = hasattr(tts, 'tts_to_file')
        has_tts_with_xtts = hasattr(tts, 'tts_with_xtts_to_file')
        has_synthesize = hasattr(tts, 'synthesize')
        
        logger.info(f"Available methods - tts: {has_tts}, tts_to_file: {has_tts_to_file}, tts_with_xtts: {has_tts_with_xtts}, synthesize: {has_synthesize}")
        
        # Get speaker information
        speakers = None
        if is_multi_speaker:
            speakers = getattr(tts, 'speakers', None)
            logger.info(f"Available speakers: {speakers}")
        
        # Get language information
        languages = None
        if is_multi_lingual:
            languages = getattr(tts, 'languages', None)
            logger.info(f"Available languages: {languages}")
            
            # Check if requested language is supported
            if languages and language not in languages:
                logger.warning(f"Requested language '{language}' not in supported languages. Using default.")
                language = languages[0] if languages else 'en'
        
        # Generate speech - try multiple methods until one works
        start_time = time.time()
        success = False
        error_messages = []
        
        # Prepare common kwargs
        common_kwargs = {}
        
        # Add language if multilingual
        if is_multi_lingual and languages and language in languages:
            common_kwargs["language"] = language
        
        # Add speaker if multi-speaker
        if is_multi_speaker and speakers and len(speakers) > 0:
            # Remove any newlines from speaker names
            clean_speakers = [s.strip() for s in speakers]
            common_kwargs["speaker"] = clean_speakers[0]
            logger.info(f"Using speaker: {clean_speakers[0]}")
        
        # Method 1: Try using tts() to get raw audio
        if has_tts and not success:
            try:
                logger.info("Trying tts() method...")
                
                # Create kwargs for this method
                kwargs = common_kwargs.copy()
                
                # Generate audio
                audio = tts.tts(text=text, **kwargs)
                
                # Save to file using soundfile
                import soundfile as sf
                sf.write(output_path, audio, 22050)  # Assuming 22050 Hz sample rate
                
                logger.info("tts() method succeeded")
                success = True
            except Exception as e:
                error_msg = f"tts() method failed: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)
        
        # Method 2: Try tts_with_xtts_to_file for XTTS models
        if has_tts_with_xtts and not success:
            try:
                logger.info("Trying tts_with_xtts_to_file method...")
                
                # Create kwargs for this method
                kwargs = common_kwargs.copy()
                kwargs["text"] = text
                kwargs["file_path"] = output_path
                
                # Call the method
                tts.tts_with_xtts_to_file(**kwargs)
                
                logger.info("tts_with_xtts_to_file method succeeded")
                success = True
            except Exception as e:
                error_msg = f"tts_with_xtts_to_file method failed: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)
        
        # Method 3: Try standard tts_to_file
        if has_tts_to_file and not success:
            try:
                logger.info("Trying tts_to_file method...")
                
                # Create kwargs for this method
                kwargs = common_kwargs.copy()
                kwargs["text"] = text
                kwargs["file_path"] = output_path
                
                # Call the method
                tts.tts_to_file(**kwargs)
                
                logger.info("tts_to_file method succeeded")
                success = True
            except Exception as e:
                error_msg = f"tts_to_file method failed: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)
        
        # Method 4: Try synthesize method
        if has_synthesize and not success:
            try:
                logger.info("Trying synthesize method...")
                
                # Create kwargs for this method
                kwargs = common_kwargs.copy()
                
                # Call the method
                audio = tts.synthesize(text, **kwargs)
                
                # Save to file using soundfile
                import soundfile as sf
                sf.write(output_path, audio, 22050)  # Assuming 22050 Hz sample rate
                
                logger.info("synthesize method succeeded")
                success = True
            except Exception as e:
                error_msg = f"synthesize method failed: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)
        
        # If all methods failed, try fallback with fixed speakers and languages
        if not success:
            logger.warning("All standard methods failed. Trying fallbacks...")
            
            # Try different speakers
            possible_speakers = ["default", "ljspeech", "p225", "p226", "p227", "male", "female", "narrator", "speaker_00", None]
            
            for speaker in possible_speakers:
                if success:
                    break
                    
                try:
                    logger.info(f"Trying with speaker: {speaker}")
                    
                    kwargs = {
                        "text": text,
                        "file_path": output_path
                    }
                    
                    if speaker:
                        kwargs["speaker"] = speaker
                    
                    if is_multi_lingual:
                        kwargs["language"] = language
                    
                    # Try to generate
                    tts.tts_to_file(**kwargs)
                    
                    # Check if file was created and not empty
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logger.info(f"Fallback with speaker '{speaker}' worked!")
                        success = True
                        break
                except Exception as e:
                    error_msg = f"Speaker '{speaker}' failed: {e}"
                    logger.warning(error_msg)
                    error_messages.append(error_msg)
        
        # Log generation time
        process_time = time.time() - start_time
        logger.info(f"TTS generation completed in {process_time:.2f} seconds")
        
        # Check if the output file was created and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Output file created: {output_path} ({file_size} bytes)")
            
            if file_size == 0:
                logger.error("Output file is empty!")
                
                # Compile all error messages
                all_errors = "; ".join(error_messages)
                return jsonify({
                    "error": "Generated audio file is empty. TTS failed.",
                    "details": all_errors
                }), 500
        else:
            logger.error(f"Output file was not created: {output_path}")
            
            # Compile all error messages
            all_errors = "; ".join(error_messages)
            return jsonify({
                "error": "Failed to generate audio file",
                "details": all_errors
            }), 500
        
        # If we got here, the file exists and has content
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="tts_speech.wav"
        )
    
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file after sending
        if 'output_path' in locals() and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")
                
@app.route('/api/whisper/tts', methods=['POST'])
def text_to_speech_whisper():
    """Convert text to speech using WhisperSpeech"""
    if not whisper_available:
        return jsonify({"error": "WhisperSpeech is not available on this server"}), 503
        
    if not request.is_json:
        return jsonify({"error": "Expected JSON request"}), 400
        
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    logger.info(f"WhisperSpeech TTS request - Text: '{text[:50]}...', Language: '{language}'")
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Generate speech
        start_time = time.time()
        
        # Use the correct parameter names as per the signature
        # fname, text, speaker=None, lang='en', cps=15, step_callback=None
        pipe.generate_to_file(
            fname=output_path,  # First parameter is fname
            text=text,          # Second parameter is text
            lang=language       # Language parameter
        )
        
        logger.info(f"WhisperSpeech generated in {time.time() - start_time:.2f} seconds")
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="speech.wav"
        )
    
    except Exception as e:
        logger.error(f"WhisperSpeech TTS failed: {e}")
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
    
    if not gtts_available:
        return jsonify({"error": "Google TTS is not available on this server"}), 503
    
    # Handle the actual POST request
    if not request.is_json:
        return jsonify({"error": "Expected JSON request"}), 400
        
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    logger.info(f"Google TTS request - Text: '{text[:50]}...', Language: '{language}'")
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_filename = temp_audio.name
        
        # Generate speech using Google TTS
        tts = gTTS(text=text, lang=language)
        tts.save(temp_filename)
        
        logger.info(f"Google TTS generated successfully")
        
        # Send file
        return send_file(
            temp_filename,
            mimetype="audio/mp3",
            as_attachment=True,
            download_name="speech.mp3"
        )
    
    except Exception as e:
        logger.error(f"Google TTS failed: {e}")
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file after sending
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

    
@app.route('/api/tts_clone', methods=['POST'])
def tts_clone():
    """Generate speech with voice cloning using XTTS"""
    if not tts_available:
        return jsonify({
            "error": "TTS voice cloning is not available. The TTS library could not be loaded or initialized properly.",
            "suggestion": "Check server logs for details on TTS initialization"
        }), 503
    
    logger.info("Received request to /api/tts_clone")
    logger.info(f"Form data keys: {list(request.form.keys())}")
    logger.info(f"Files keys: {list(request.files.keys())}")
    
    # Check if this is a proper multipart/form-data request
    if 'voice_file' not in request.files:
        return jsonify({"error": "No voice reference file provided"}), 400
    
    if 'text' not in request.form:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.form.get('text')
    language = request.form.get('language', 'en')
    logger.info(f"TTS clone request - Text: '{text[:50]}...', Language: '{language}'")
    
    # Save reference voice file
    voice_file = request.files['voice_file']
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_voice:
        voice_path = temp_voice.name
        voice_file.save(voice_path)
    
    try:
        # Create temporary file for output audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            output_path = temp_audio.name
        
        # Determine which synthesis method to use
        start_time = time.time()
        model_name = getattr(tts, 'model_name', 'unknown')
        logger.info(f"Using model: {model_name}")
        
        # Check if the model has the XTTS-specific method
        if hasattr(tts, 'tts_with_xtts_to_file'):
            logger.info("Using tts_with_xtts_to_file method for voice cloning")
            
            # Prepare arguments
            kwargs = {
                "text": text,
                "speaker_wav": voice_path,
                "file_path": output_path
            }
            
            # Only add language parameter if the model is multilingual
            if tts_is_multilingual:
                kwargs["language"] = language
                logger.info(f"Using language: {language}")
            
            # Call the method
            tts.tts_with_xtts_to_file(**kwargs)
            
        elif hasattr(tts, 'tts_to_file'):
            logger.info("Using standard tts_to_file method for voice cloning")
            
            # Prepare arguments
            kwargs = {
                "text": text,
                "speaker_wav": voice_path,
                "file_path": output_path
            }
            
            # Only add language parameter if the model is multilingual
            if tts_is_multilingual:
                kwargs["language"] = language
                logger.info(f"Using language: {language}")
            
            # Call the method
            tts.tts_to_file(**kwargs)
            
        else:
            logger.error("No suitable method found for voice cloning")
            return jsonify({"error": "The loaded TTS model does not support voice cloning"}), 500
        
        logger.info(f"Voice cloning completed in {time.time() - start_time:.2f} seconds")
        
        # Check if the output file was created and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Output file created: {output_path} ({file_size} bytes)")
            
            if file_size == 0:
                logger.error("Output file is empty!")
                return jsonify({"error": "Generated audio file is empty. Voice cloning failed."}), 500
        else:
            logger.error(f"Output file was not created: {output_path}")
            return jsonify({"error": "Failed to generate audio file"}), 500
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="cloned_speech.wav"
        )
    
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Voice cloning failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp voice file: {e}")

@app.route('/api/whisper/tts_clone', methods=['POST'])
def tts_clone_whisper():
    """Generate speech with voice cloning using WhisperSpeech"""
    if not whisper_available:
        return jsonify({"error": "WhisperSpeech is not available on this server"}), 503
    
    # Check if this is a proper multipart/form-data request
    if 'voice_file' not in request.files:
        return jsonify({"error": "No voice reference file provided"}), 400
    
    if 'text' not in request.form:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.form.get('text')
    language = request.form.get('language', 'en')
    logger.info(f"WhisperSpeech clone request - Text: '{text[:50]}...', Language: '{language}'")
    
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
        pipe.generate_to_file(
            fname=output_path,      # First parameter is fname
            text=text,              # Second parameter is text
            speaker=voice_path,     # Use speaker parameter for voice cloning
            lang=language           # Language parameter
        )
        
        logger.info(f"WhisperSpeech clone generated in {time.time() - start_time:.2f} seconds")
        
        # Send file
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="cloned_speech.wav"
        )
    
    except Exception as e:
        logger.error(f"WhisperSpeech clone failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp voice file: {e}")

@app.route('/api/styletts/tts_clone', methods=['POST'])
def tts_styletts2():
    """Generate speech with StyleTTS2"""
    return jsonify({"error": "StyleTTS2 is not available in this version"}), 503

# Add a cleanup route to manually delete temporary files if needed
@app.route('/api/cleanup', methods=['POST'])
def cleanup_temp_files():
    """Clean up temporary files in the temp directory"""
    temp_dir = tempfile.gettempdir()
    count = 0
    
    try:
        for filename in os.listdir(temp_dir):
            if (filename.endswith('.mp3') or filename.endswith('.wav')) and os.path.isfile(os.path.join(temp_dir, filename)):
                try:
                    os.unlink(os.path.join(temp_dir, filename))
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {filename}: {e}")
        
        return jsonify({"success": True, "message": f"Cleaned up {count} temporary files"})
    
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the status of available TTS services"""
    return jsonify({
        "whisper_tts": whisper_available,
        "google_tts": gtts_available,
        "xtts_clone": tts_available,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)