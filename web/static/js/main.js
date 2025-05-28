document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const modelSelect = document.getElementById('modelSelect');
    const changeModelBtn = document.getElementById('changeModelBtn');
    const systemInfo = document.getElementById('systemInfo');
    const modelStatus = document.getElementById('modelStatus');
    const audioFile = document.getElementById('audioFile');
    const languageSelect = document.getElementById('languageSelect');
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const transcribeBtn = document.getElementById('transcribeBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultContainer = document.getElementById('resultContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    const transcriptionResult = document.getElementById('transcriptionResult');
    const resultModel = document.getElementById('resultModel');
    const resultLanguage = document.getElementById('resultLanguage');
    const resultTime = document.getElementById('resultTime');
    const resultDevice = document.getElementById('resultDevice');
    const recordingContainer = document.getElementById('recordingContainer');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingProgress = document.getElementById('recordingProgress');
	const generateGoogleSpeechBtn = document.getElementById('generateGoogleSpeechBtn');
	const generateXTTSSpeechBtn = document.getElementById('generateXTTSSpeechBtn');
	const generateWhisperSpeechBtn = document.getElementById('generateWhisperSpeechBtn');
	
	const generateXTTSClonedSpeechBtn = document.getElementById('generateXTTSClonedSpeechBtn');
    const generateStyleTTSClonedSpeechBtn = document.getElementById('generateStyleTTSClonedSpeechBtn');
	const generateWhisperClonedSpeechBtn = document.getElementById('generateWhisperClonedSpeechBtn');
	
    // MediaRecorder variables
    let mediaRecorder;
    let audioChunks = [];
    let recordingInterval;
    let recordingSeconds = 0;
    let maxRecordingTime = 300; // 5 minutes max
	
	const speedControl = document.getElementById('ttsSpeedControl');
    const speedValue = document.getElementById('speedValue');
	
	if (speedControl && speedValue) {
        speedControl.addEventListener('input', function() {
            speedValue.textContent = speedControl.value;
        });
    }
    
    // Add event listener for the main generate button
    const generateSpeechBtn = document.getElementById('generateSpeechBtn');
    if (generateSpeechBtn) {
        generateSpeechBtn.addEventListener('click', generateSpeech);
    }
    // Fetch available models and system info
    fetchModels();

    // Event listeners
    changeModelBtn.addEventListener('click', changeModel);
    transcribeBtn.addEventListener('click', transcribeAudio);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
	
	if (generateGoogleSpeechBtn) {
        generateGoogleSpeechBtn.addEventListener('click', generateGoogleSpeech);
    }
	if (generateXTTSSpeechBtn) {
        generateXTTSSpeechBtn.addEventListener('click', generateXTTSSpeech);
    }
	if (generateWhisperSpeechBtn) {
        generateWhisperSpeechBtn.addEventListener('click', generateWhisperSpeech);
    }
	if (generateXTTSClonedSpeechBtn) {
        generateXTTSClonedSpeechBtn.addEventListener('click', generateXTTSClonedSpeech);
    }
	if (generateStyleTTSClonedSpeechBtn) {
        generateStyleTTSClonedSpeechBtn.addEventListener('click', generateStyleTTSClonedSpeech);
    }
	if (generateWhisperClonedSpeechBtn) {
        generateWhisperClonedSpeechBtn.addEventListener('click', generateWhisperClonedSpeech);
    }
    // Functions
    function fetchModels() {
        fetch('/whisper-stt/api/models')
            .then(response => response.json())
            .then(data => {
                // Set the current model in the select dropdown
                modelSelect.value = data.current;
                
                // Update system info
                systemInfo.textContent = `Currently using ${data.current} model on ${data.device}`;
            })
            .catch(error => {
                console.error('Error fetching models:', error);
                systemInfo.textContent = 'Error fetching system information';
            });
    }

    function changeModel() {
        const selectedModel = modelSelect.value;
        modelStatus.textContent = 'Loading model...';
        
        fetch('/whisper-stt/api/change_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: selectedModel })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                modelStatus.textContent = `Model changed to ${data.model} (loaded in ${data.load_time})`;
                systemInfo.textContent = `Currently using ${data.model} model`;
                
                // Clear status after 5 seconds
                setTimeout(() => {
                    modelStatus.textContent = '';
                }, 5000);
            } else {
                modelStatus.textContent = data.error || 'Failed to change model';
            }
        })
        .catch(error => {
            console.error('Error changing model:', error);
            modelStatus.textContent = 'Error changing model';
        });
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            recordingSeconds = 0;
            
            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });
            
            mediaRecorder.addEventListener('stop', () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayer.src = audioUrl;
                
                // Create a file from the blob for the API call
                const file = new File([audioBlob], "recording.wav", { type: 'audio/wav' });
                
                // Set the file to the file input
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                audioFile.files = dataTransfer.files;
                
                // Reset recording UI
                recordBtn.classList.remove('recording');
                recordBtn.disabled = false;
                stopBtn.disabled = true;
                recordingContainer.style.display = 'none';
                
                // Clear the recording interval
                clearInterval(recordingInterval);
            });
            
            // Start recording
            mediaRecorder.start();
            
            // Update UI
            recordBtn.classList.add('recording');
            recordBtn.disabled = true;
            stopBtn.disabled = false;
            recordingContainer.style.display = 'block';
            
            // Set up recording timer
            recordingInterval = setInterval(() => {
                recordingSeconds++;
                const minutes = Math.floor(recordingSeconds / 60);
                const seconds = recordingSeconds % 60;
                recordingStatus.textContent = `Recording: ${minutes}:${seconds.toString().padStart(2, '0')}`;
                
                // Update progress bar
                const progress = (recordingSeconds / maxRecordingTime) * 100;
                recordingProgress.style.width = `${Math.min(progress, 100)}%`;
                
                // Auto-stop if max recording time is reached
                if (recordingSeconds >= maxRecordingTime) {
                    stopRecording();
                }
            }, 1000);
            
        } catch (error) {
            console.error('Error starting recording:', error);
            alert('Could not access microphone. Please ensure you have granted permission.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    function transcribeAudio() {
        if (!audioFile.files.length) {
            alert('Please select an audio file or record audio first');
            return;
        }
        
        const file = audioFile.files[0];
        const language = languageSelect.value;
        
        // Create form data
        const formData = new FormData();
        formData.append('audio', file);
        if (language) {
            formData.append('language', language);
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultContainer.style.display = 'none';
        
        // Send request
        fetch('/whisper-stt/api/transcribe', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Hide loading indicator
                loadingIndicator.style.display = 'none';
                resultContainer.style.display = 'block';
                
                // Update result UI
                transcriptionResult.value = data.text;
                resultModel.textContent = data.model;
                resultLanguage.textContent = data.language;
                resultTime.textContent = data.process_time;
                resultDevice.textContent = data.device;
                
                // Set audio player source if it's a file upload (not a recording)
                if (data.file_url && !audioPlayer.src) {
                    audioPlayer.src = data.file_url;
                }
            } else {
                alert('Transcription failed: ' + (data.error || 'Unknown error'));
                loadingIndicator.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error transcribing audio:', error);
            alert('Error transcribing audio. Please try again.');
            loadingIndicator.style.display = 'none';
        });
    }
	
	async function generateWhisperSpeech() {
		const textArea = document.getElementById('textToSpeech');
		const languageSelect = document.getElementById('ttsLanguageSelect');
		const audioPlayer = document.getElementById('ttsAudioPlayer');
		
		if (!textArea || !languageSelect || !audioPlayer) {
			console.error('Required elements not found');
			return;
		}
		
		const text = textArea.value.trim();
		if (!text) {
			alert('Please enter some text to convert to speech');
			return;
		}
		
		const language = languageSelect.value;
		
		try {
			// Show loading state
			const generateWhisperSpeechBtn = document.getElementById('generateWhisperSpeechBtn');
			if (generateWhisperSpeechBtn) {
				generateWhisperSpeechBtn.disabled = true;
				generateWhisperSpeechBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
			}
			
			console.log('Sending TTS request:', { text, language });
			
			// Send request to the server
			const response = await fetch('/whisper-stt/api/whisper/tts', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Accept': 'audio/mpeg, audio/mp3'
				},
				body: JSON.stringify({
					text: text,
					language: language
				})
			});
			
			// Reset button state
			if (generateWhisperSpeechBtn) {
				generateWhisperSpeechBtn.disabled = false;
				generateWhisperSpeechBtn.textContent = 'Generate Whisper Speech';
			}
			
			console.log('TTS response status:', response.status);
			console.log('TTS response headers:', response.headers);
			
			if (!response.ok) {
				if (response.headers.get('content-type')?.includes('application/json')) {
					const errorData = await response.json();
					throw new Error(errorData.error || `HTTP error ${response.status}`);
				} else {
					throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
				}
			}
			
			// Get audio blob and play it
			const audioBlob = await response.blob();
			console.log('Received audio blob:', audioBlob);
			
			const audioUrl = URL.createObjectURL(audioBlob);
			audioPlayer.src = audioUrl;
			audioPlayer.play();
			
		} catch (error) {
			console.error('Error generating speech:', error);
			alert('Error generating speech: ' + error.message);
			
			// Reset button state in case of error
			const generateWhisperSpeechBtn = document.getElementById('generateWhisperSpeechBtn');
			if (generateWhisperSpeechBtn) {
				generateWhisperSpeechBtn.disabled = false;
				generateWhisperSpeechBtn.textContent = 'Generate Whisper Speech';
			}
		}
	}
	
	async function generateGoogleSpeech() {
		const textArea = document.getElementById('textToSpeech');
		const languageSelect = document.getElementById('ttsLanguageSelect');
		const audioPlayer = document.getElementById('ttsAudioPlayer');
		
		if (!textArea || !languageSelect || !audioPlayer) {
			console.error('Required elements not found');
			return;
		}
		
		const text = textArea.value.trim();
		if (!text) {
			alert('Please enter some text to convert to speech');
			return;
		}
		
		const language = languageSelect.value;
		
		try {
			// Show loading state
			const generateGoogleSpeechBtn = document.getElementById('generateGoogleSpeechBtn');
			if (generateGoogleSpeechBtn) {
				generateGoogleSpeechBtn.disabled = true;
				generateGoogleSpeechBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
			}
			
			console.log('Sending TTS request:', { text, language });
			
			// Send request to the server
			const response = await fetch('/whisper-stt/api/google/tts', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Accept': 'audio/mpeg, audio/mp3'
				},
				body: JSON.stringify({
					text: text,
					language: language
				})
			});
			
			// Reset button state
			if (generateGoogleSpeechBtn) {
				generateGoogleSpeechBtn.disabled = false;
				generateGoogleSpeechBtn.textContent = 'Generate Google Speech';
			}
			
			console.log('TTS response status:', response.status);
			console.log('TTS response headers:', response.headers);
			
			if (!response.ok) {
				if (response.headers.get('content-type')?.includes('application/json')) {
					const errorData = await response.json();
					throw new Error(errorData.error || `HTTP error ${response.status}`);
				} else {
					throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
				}
			}
			
			// Get audio blob and play it
			const audioBlob = await response.blob();
			console.log('Received audio blob:', audioBlob);
			
			const audioUrl = URL.createObjectURL(audioBlob);
			audioPlayer.src = audioUrl;
			audioPlayer.play();
			
		} catch (error) {
			console.error('Error generating speech:', error);
			alert('Error generating speech: ' + error.message);
			
			// Reset button state in case of error
			const generateGoogleSpeechBtn = document.getElementById('generateGoogleSpeechBtn');
			if (generateGoogleSpeechBtn) {
				generateGoogleSpeechBtn.disabled = false;
				generateGoogleSpeechBtn.textContent = 'Generate Google Speech';
			}
		}
	}
	
	async function generateXTTSSpeech() {
		const textArea = document.getElementById('textToSpeech');
		const languageSelect = document.getElementById('ttsLanguageSelect');
		const audioPlayer = document.getElementById('ttsAudioPlayer');
		
		if (!textArea || !languageSelect || !audioPlayer) {
			console.error('Required elements not found');
			return;
		}
		
		const text = textArea.value.trim();
		if (!text) {
			alert('Please enter some text to convert to speech');
			return;
		}
		
		const language = languageSelect.value;
		
		try {
			// Show loading state
			const generateXTTSSpeechBtn = document.getElementById('generateXTTSSpeechBtn');
			if (generateXTTSSpeechBtn) {
				generateXTTSSpeechBtn.disabled = true;
				generateXTTSSpeechBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
			}
			
			console.log('Sending XTTS request:', { text, language });
			
			// Send request to the server
			const response = await fetch('/whisper-stt/api/xtts/tts', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Accept': 'audio/wav'
				},
				body: JSON.stringify({
					text: text,
					language: language
				})
			});
			
			// Reset button state
			if (generateXTTSSpeechBtn) {
				generateXTTSSpeechBtn.disabled = false;
				generateXTTSSpeechBtn.textContent = 'Generate XTTS Speech';
			}
			
			console.log('XTTS response status:', response.status);
			console.log('XTTS response headers:', response.headers);
			
			if (!response.ok) {
				if (response.headers.get('content-type')?.includes('application/json')) {
					const errorData = await response.json();
					throw new Error(errorData.error || `HTTP error ${response.status}`);
				} else {
					throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
				}
			}
			
			// Get audio blob and play it
			const audioBlob = await response.blob();
			console.log('Received audio blob:', audioBlob);
			
			const audioUrl = URL.createObjectURL(audioBlob);
			audioPlayer.src = audioUrl;
			audioPlayer.play();
			
		} catch (error) {
			console.error('Error generating XTTS speech:', error);
			alert('Error generating XTTS speech: ' + error.message);
			
			// Reset button state in case of error
			const generateXTTSSpeechBtn = document.getElementById('generateXTTSSpeechBtn');
			if (generateXTTSSpeechBtn) {
				generateXTTSSpeechBtn.disabled = false;
				generateXTTSSpeechBtn.textContent = 'Generate XTTS Speech';
			}
		}
	}
	
	async function generateXTTSClonedSpeech() {
		const textArea = document.getElementById('cloneText');
		const voiceFileInput = document.getElementById('voiceFile');
		const languageSelect = document.getElementById('cloneLanguageSelect');
		const audioPlayer = document.getElementById('clonedAudioPlayer');
		
		if (!textArea || !voiceFileInput || !languageSelect || !audioPlayer) {
			console.error('Required elements not found');
			return;
		}
		
		const text = textArea.value.trim();
		if (!text) {
			alert('Please enter some text to convert to speech');
			return;
		}
		
		if (!voiceFileInput.files || voiceFileInput.files.length === 0) {
			alert('Please upload a reference voice file');
			return;
		}
		
		const voiceFile = voiceFileInput.files[0];
		const language = languageSelect.value;
		
		try {
			// Show loading state
			const generateXTTSClonedSpeechBtn = document.getElementById('generateXTTSClonedSpeechBtn');
			if (generateXTTSClonedSpeechBtn) {
				generateXTTSClonedSpeechBtn.disabled = true;
				generateXTTSClonedSpeechBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
			}
			
			// Create form data
			const formData = new FormData();
			formData.append('text', text);
			formData.append('language', language);
			formData.append('voice_file', voiceFile);
			
			// Send request to the server
			const response = await fetch('/whisper-stt/api/tts_clone', {
				method: 'POST',
				body: formData  // Use only this formData object with all params
			});
			
			// Reset button state
			if (generateXTTSClonedSpeechBtn) {
				generateXTTSClonedSpeechBtn.disabled = false;
				generateXTTSClonedSpeechBtn.textContent = 'Generate XTTS Cloned Speech';
			}
			
			if (!response.ok) {
				if (response.headers.get('content-type')?.includes('application/json')) {
					const errorData = await response.json();
					throw new Error(errorData.error || `HTTP error ${response.status}`);
				} else {
					throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
				}
			}
			
			// Get audio blob and play it
			const audioBlob = await response.blob();
			const audioUrl = URL.createObjectURL(audioBlob);
			
			audioPlayer.src = audioUrl;
			audioPlayer.play();
			
		} catch (error) {
			console.error('Error generating cloned speech:', error);
			alert('Error generating cloned speech: ' + error.message);
			
			// Reset button state in case of error
			const generateXTTSClonedSpeechBtn = document.getElementById('generateXTTSClonedSpeechBtn');
			if (generateXTTSClonedSpeechBtn) {
				generateXTTSClonedSpeechBtn.disabled = false;
				generateXTTSClonedSpeechBtn.textContent = 'Generate XTTS Cloned Speech';
			}
		}
	}

	// Fixed generateStyleTTSClonedSpeech function
	async function generateStyleTTSClonedSpeech() {
		const textArea = document.getElementById('cloneText');
		const voiceFileInput = document.getElementById('voiceFile');
		const languageSelect = document.getElementById('cloneLanguageSelect');
		const audioPlayer = document.getElementById('clonedAudioPlayer');
		
		if (!textArea || !voiceFileInput || !languageSelect || !audioPlayer) {
			console.error('Required elements not found');
			return;
		}
		
		const text = textArea.value.trim();
		if (!text) {
			alert('Please enter some text to convert to speech');
			return;
		}
		
		if (!voiceFileInput.files || voiceFileInput.files.length === 0) {
			alert('Please upload a reference voice file');
			return;
		}
		
		const voiceFile = voiceFileInput.files[0];
		const language = languageSelect.value;
		
		try {
			// Show loading state
			const generateStyleTTSClonedSpeechBtn = document.getElementById('generateStyleTTSClonedSpeechBtn');
			if (generateStyleTTSClonedSpeechBtn) {
				generateStyleTTSClonedSpeechBtn.disabled = true;
				generateStyleTTSClonedSpeechBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
			}
			
			// Create form data
			const formData = new FormData();
			formData.append('text', text);
			formData.append('language', language);
			formData.append('voice_file', voiceFile);
			
			// Send request to the server
			const response = await fetch('/whisper-stt/api/styletts/tts_clone', {
				method: 'POST',
				body: formData
			});
			
			// Reset button state
			if (generateStyleTTSClonedSpeechBtn) {
				generateStyleTTSClonedSpeechBtn.disabled = false;
				generateStyleTTSClonedSpeechBtn.textContent = 'Generate StyleTTS Cloned Speech';
			}
			
			if (!response.ok) {
				if (response.headers.get('content-type')?.includes('application/json')) {
					const errorData = await response.json();
					throw new Error(errorData.error || `HTTP error ${response.status}`);
				} else {
					throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
				}
			}
			
			// Get audio blob and play it
			const audioBlob = await response.blob();
			const audioUrl = URL.createObjectURL(audioBlob);
			
			audioPlayer.src = audioUrl;
			audioPlayer.play();
			
		} catch (error) {
			console.error('Error generating cloned speech:', error);
			alert('Error generating cloned speech: ' + error.message);
			
			// Reset button state in case of error
			const generateStyleTTSClonedSpeechBtn = document.getElementById('generateStyleTTSClonedSpeechBtn');
			if (generateStyleTTSClonedSpeechBtn) {
				generateStyleTTSClonedSpeechBtn.disabled = false;
				generateStyleTTSClonedSpeechBtn.textContent = 'Generate StyleTTS Cloned Speech';
			}
		}
	}

	// Fixed generateWhisperClonedSpeech function
	async function generateWhisperClonedSpeech() {
		const textArea = document.getElementById('cloneText');
		const voiceFileInput = document.getElementById('voiceFile');
		const languageSelect = document.getElementById('cloneLanguageSelect');
		const audioPlayer = document.getElementById('clonedAudioPlayer');
		
		if (!textArea || !voiceFileInput || !languageSelect || !audioPlayer) {
			console.error('Required elements not found');
			return;
		}
		
		const text = textArea.value.trim();
		if (!text) {
			alert('Please enter some text to convert to speech');
			return;
		}
		
		if (!voiceFileInput.files || voiceFileInput.files.length === 0) {
			alert('Please upload a reference voice file');
			return;
		}
		
		const voiceFile = voiceFileInput.files[0];
		const language = languageSelect.value;
		
		try {
			// Show loading state
			const generateWhisperClonedSpeechBtn = document.getElementById('generateWhisperClonedSpeechBtn');
			if (generateWhisperClonedSpeechBtn) {
				generateWhisperClonedSpeechBtn.disabled = true;
				generateWhisperClonedSpeechBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
			}
			
			// Create form data
			const formData = new FormData();
			formData.append('text', text);
			formData.append('language', language);
			formData.append('voice_file', voiceFile);
			
			// Send request to the server
			const response = await fetch('/whisper-stt/api/whisper/tts_clone', {
				method: 'POST',
				body: formData
			});
			
			// Reset button state
			if (generateWhisperClonedSpeechBtn) {
				generateWhisperClonedSpeechBtn.disabled = false;
				generateWhisperClonedSpeechBtn.textContent = 'Generate Whisper Cloned Speech';
			}
			
			if (!response.ok) {
				if (response.headers.get('content-type')?.includes('application/json')) {
					const errorData = await response.json();
					throw new Error(errorData.error || `HTTP error ${response.status}`);
				} else {
					throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
				}
			}
			
			// Get audio blob and play it
			const audioBlob = await response.blob();
			const audioUrl = URL.createObjectURL(audioBlob);
			
			audioPlayer.src = audioUrl;
			audioPlayer.play();
			
		} catch (error) {
			console.error('Error generating cloned speech:', error);
			alert('Error generating cloned speech: ' + error.message);
			
			// Reset button state in case of error
			const generateWhisperClonedSpeechBtn = document.getElementById('generateWhisperClonedSpeechBtn');
			if (generateWhisperClonedSpeechBtn) {
				generateWhisperClonedSpeechBtn.disabled = false;
				generateWhisperClonedSpeechBtn.textContent = 'Generate Whisper Cloned Speech';
			}
		}
	}
});