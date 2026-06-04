document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------
    // 1. DYNAMIC FILE INPUT LABEL (for detect.html and signup.html)
    // -------------------------------------------------------------------
    const fileInputs = document.querySelectorAll('.tech-file-input');
    fileInputs.forEach(input => {
        input.addEventListener('change', (event) => {
            const fileInfo = input.closest('.form-group').querySelector('.file-info');
            if (fileInfo) {
                if (input.files.length > 0) {
                    // Display the name of the selected file
                    fileInfo.textContent = `File Selected: ${input.files[0].name}`;
                } else {
                    // Reset to default text
                    fileInfo.textContent = 'No file selected.'; 
                }
            }
        });
    });


    // -------------------------------------------------------------------
    // 2. DEEPFAKE DETECTION LOGIC (for detect.html)
    // -------------------------------------------------------------------
    const detectionForm = document.getElementById('detection-form');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsPanel = document.getElementById('results-panel');
    const resultPrediction = document.getElementById('result-prediction');
    const resultConfidence = document.getElementById('result-confidence');
    const confidenceBarInner = document.getElementById('confidence-bar-inner');
    const submitBtn = document.getElementById('submit-btn');

    if (detectionForm) {
        detectionForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const videoFile = document.getElementById('video_file').files[0];
            if (!videoFile) {
                alert("Please select a video file first.");
                return;
            }

            // --- A. Prepare UI for Submission ---
            resultsPanel.style.display = 'none'; // Hide old results
            loadingIndicator.style.display = 'block'; // Show loading spinner
            submitBtn.disabled = true;
            submitBtn.querySelector('.btn-text').textContent = 'SCANNING...';

            const formData = new FormData();
            formData.append('video_file', videoFile);

            // --- B. Send AJAX Request to Flask Endpoint ---
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                    // Flask handles the CSRF token via cookies, so headers are minimal
                });

                // Check for HTTP errors (e.g., 400, 500)
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                }

                const data = await response.json();

                // --- C. Process Results and Update UI ---
                if (data.success) {
                    const confidence = parseFloat(data.confidence); // e.g., 0.98
                    const percentage = (confidence * 100).toFixed(2);
                    const prediction = data.prediction; // 'REAL' or 'FAKE'
                    
                    // 1. Update text and color
                    resultPrediction.textContent = prediction;
                    resultConfidence.textContent = `${percentage}%`;
                    
                    // Clear previous classes
                    resultPrediction.classList.remove('real', 'fake'); 
                    
                    if (prediction === 'FAKE') {
                        resultPrediction.classList.add('fake');
                        // High confidence in Fake -> Pink bar
                        confidenceBarInner.style.backgroundColor = 'var(--neon-pink)'; 
                    } else {
                        resultPrediction.classList.add('real');
                        // High confidence in Real -> Blue bar
                        confidenceBarInner.style.backgroundColor = 'var(--neon-blue)';
                    }
                    
                    // 2. Update visual confidence bar
                    // If prediction is FAKE, the bar should represent FAKE confidence (e.g., 98% fake)
                    // If prediction is REAL, the bar should represent REAL confidence (e.g., 98% real)
                    confidenceBarInner.style.width = `${percentage}%`;

                } else {
                    // Handle API-side error (e.g., "No face detected", "Invalid file")
                    throw new Error(data.message || "Detection failed due to an unknown issue.");
                }

            } catch (error) {
                // Handle network or processing errors
                resultPrediction.textContent = 'ERROR';
                resultPrediction.classList.add('fake');
                resultConfidence.textContent = 'Check console.';
                console.error("Detection Error:", error);
                alert(`Error during detection: ${error.message}`);
                confidenceBarInner.style.width = '100%';
                confidenceBarInner.style.backgroundColor = 'var(--neon-yellow)'; // Yellow for system error
                resultsPanel.style.display = 'block'; 

            } finally {
                // --- D. Finalize UI ---
                loadingIndicator.style.display = 'none';
                submitBtn.disabled = false;
                submitBtn.querySelector('.btn-text').textContent = 'SCAN VIDEO FILE';
                // Show results only if no fatal error occurred (check if error was already displayed)
                if (resultPrediction.textContent !== 'ERROR') {
                    resultsPanel.style.display = 'block';
                }
            }
        });
    }

    // You can add more general JavaScript functions here (e.g., dark mode toggle, menu animation)
});