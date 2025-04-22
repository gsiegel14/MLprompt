document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const systemPromptTextarea = document.getElementById('system-prompt');
    const outputPromptTextarea = document.getElementById('output-prompt');
    const examplesTextarea = document.getElementById('examples-text');
    const csvFileInput = document.getElementById('csv-file');
    const runButton = document.getElementById('run-button');
    const saveButton = document.getElementById('save-button');
    const resultsTable = document.getElementById('results-table');
    const resultsBody = document.getElementById('results-body');
    const resultsSection = document.getElementById('results-section');
    const spinnerElement = document.getElementById('spinner');
    const alertContainer = document.getElementById('alert-container');
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Show sample prompts
    document.getElementById('show-sample-prompts').addEventListener('click', function() {
        systemPromptTextarea.value = 
`You are an expert assistant helping users with their questions.
Always provide accurate, helpful, and concise responses.
Be polite and professional at all times.`;

        outputPromptTextarea.value = 
`Please answer the following question:
[Question]
{user_input}
[/Question]

Your answer should be clear, accurate, and helpful.`;

        examplesTextarea.value = 
`What is the capital of France?,Paris
How many planets are in our solar system?,Eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune
What's the square root of 64?,8`;
    });

    // Clear results
    function clearResults() {
        resultsBody.innerHTML = '';
        resultsSection.style.display = 'none';
    }

    // Show alert message
    function showAlert(message, type = 'danger') {
        alertContainer.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = document.querySelector('.alert');
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }

    // Handle CSV file upload
    csvFileInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            
            // Check file type
            if (!file.name.endsWith('.csv')) {
                showAlert('Please upload a CSV file');
                csvFileInput.value = '';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            // Show spinner
            spinnerElement.style.display = 'block';
            
            fetch('/upload_csv', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error);
                } else if (data.examples && data.examples.length > 0) {
                    // Convert examples to CSV format and fill textarea
                    let csvContent = '';
                    data.examples.forEach(example => {
                        csvContent += `${example.user_input},${example.ground_truth_output}\n`;
                    });
                    examplesTextarea.value = csvContent.trim();
                    showAlert(`Successfully loaded ${data.examples.length} examples from CSV`, 'success');
                } else {
                    showAlert('No valid examples found in the CSV file');
                }
            })
            .catch(error => {
                showAlert('Error uploading file: ' + error.message);
            })
            .finally(() => {
                spinnerElement.style.display = 'none';
                csvFileInput.value = ''; // Reset file input
            });
        }
    });

    // Run evaluation button handler
    runButton.addEventListener('click', function() {
        const systemPrompt = systemPromptTextarea.value.trim();
        const outputPrompt = outputPromptTextarea.value.trim();
        const examplesText = examplesTextarea.value.trim();
        
        // Validate inputs
        if (!systemPrompt) {
            showAlert('System prompt is required');
            return;
        }
        
        if (!outputPrompt) {
            showAlert('Output prompt is required');
            return;
        }
        
        if (!examplesText) {
            showAlert('Example data is required');
            return;
        }
        
        // Prepare data
        const data = {
            system_prompt: systemPrompt,
            output_prompt: outputPrompt,
            examples_format: 'text',
            examples_content: examplesText
        };
        
        // Clear previous results
        clearResults();
        
        // Show spinner
        spinnerElement.style.display = 'block';
        runButton.disabled = true;
        
        // Call API
        fetch('/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error);
            } else if (data.results && data.results.length > 0) {
                // Display results
                data.results.forEach((result, index) => {
                    const scoreClass = getScoreClass(result.score);
                    const scoreText = getScoreText(result.score);
                    
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><pre>${index + 1}</pre></td>
                        <td><pre>${escapeHtml(result.user_input)}</pre></td>
                        <td><pre>${escapeHtml(result.ground_truth_output)}</pre></td>
                        <td><pre>${escapeHtml(result.model_response)}</pre></td>
                        <td class="${scoreClass}">${scoreText} (${result.score.toFixed(2)})</td>
                    `;
                    resultsBody.appendChild(row);
                });
                
                resultsSection.style.display = 'block';
                
                // Scroll to results
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                showAlert('No results returned from evaluation');
            }
        })
        .catch(error => {
            showAlert('Error running evaluation: ' + error.message);
        })
        .finally(() => {
            spinnerElement.style.display = 'none';
            runButton.disabled = false;
        });
    });
    
    // Save prompts button handler
    saveButton.addEventListener('click', function() {
        const systemPrompt = systemPromptTextarea.value.trim();
        const outputPrompt = outputPromptTextarea.value.trim();
        
        // Validate inputs
        if (!systemPrompt) {
            showAlert('System prompt is required');
            return;
        }
        
        if (!outputPrompt) {
            showAlert('Output prompt is required');
            return;
        }
        
        // Prepare data
        const data = {
            system_prompt: systemPrompt,
            output_prompt: outputPrompt
        };
        
        // Show spinner
        spinnerElement.style.display = 'block';
        saveButton.disabled = true;
        
        // Call API
        fetch('/save_prompts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error);
            } else {
                showAlert('Prompts saved successfully as ' + data.system_filename + ' and ' + data.output_filename, 'success');
            }
        })
        .catch(error => {
            showAlert('Error saving prompts: ' + error.message);
        })
        .finally(() => {
            spinnerElement.style.display = 'none';
            saveButton.disabled = false;
        });
    });
    
    // Helper functions
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function getScoreClass(score) {
        if (score >= 0.9) return 'score-perfect';
        if (score > 0) return 'score-partial';
        return 'score-none';
    }
    
    function getScoreText(score) {
        if (score >= 0.9) return 'Perfect Match';
        if (score > 0.5) return 'Good Match';
        if (score > 0) return 'Partial Match';
        return 'No Match';
    }
});
