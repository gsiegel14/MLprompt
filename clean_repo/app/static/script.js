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
    
    // Performance overview elements
    const avgScoreElement = document.getElementById('avg-score');
    const perfectMatchesElement = document.getElementById('perfect-matches');
    const totalExamplesElement = document.getElementById('total-examples');
    
    // Add elegant entrance animations
    document.querySelectorAll('.card').forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
    
    // Initialize tooltips with modern styling
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            boundary: document.body
        });
    });

    // Show sample prompts with fade-in effect
    document.getElementById('show-sample-prompts').addEventListener('click', function() {
        // Apply fade-out effect
        systemPromptTextarea.style.opacity = '0.3';
        outputPromptTextarea.style.opacity = '0.3';
        examplesTextarea.style.opacity = '0.3';
        
        setTimeout(() => {
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

            // Apply fade-in effect
            systemPromptTextarea.style.opacity = '1';
            outputPromptTextarea.style.opacity = '1';
            examplesTextarea.style.opacity = '1';
        }, 300);
    });

    // Clear results with animation
    function clearResults() {
        resultsBody.innerHTML = '';
        avgScoreElement.textContent = '-';
        perfectMatchesElement.textContent = '-';
        totalExamplesElement.textContent = '-';
        
        if (resultsSection.style.display !== 'none') {
            resultsSection.style.opacity = '0';
            setTimeout(() => {
                resultsSection.style.display = 'none';
                resultsSection.style.opacity = '1';
            }, 300);
        }
    }

    // Show alert message with improved animation
    function showAlert(message, type = 'danger') {
        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${type} alert-dismissible fade show`;
        alertElement.setAttribute('role', 'alert');
        alertElement.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Clear previous alerts
        alertContainer.innerHTML = '';
        alertContainer.appendChild(alertElement);
        
        // Apply entrance animation
        alertElement.style.animationName = 'fadeInDown';
        alertElement.style.animationDuration = '0.5s';
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alertElement);
            bsAlert.close();
        }, 5000);
    }

    // Handle CSV file upload with improved UX
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
            
            // Show spinner with fade-in
            spinnerElement.style.opacity = '0';
            spinnerElement.style.display = 'block';
            setTimeout(() => {
                spinnerElement.style.opacity = '1';
            }, 10);
            
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
                    
                    // Apply fade effect for textarea update
                    examplesTextarea.style.opacity = '0.3';
                    setTimeout(() => {
                        examplesTextarea.value = csvContent.trim();
                        examplesTextarea.style.opacity = '1';
                    }, 300);
                    
                    showAlert(`Successfully loaded ${data.examples.length} examples from CSV`, 'success');
                } else {
                    showAlert('No valid examples found in the CSV file');
                }
            })
            .catch(error => {
                showAlert('Error uploading file: ' + error.message);
            })
            .finally(() => {
                // Hide spinner with fade-out
                spinnerElement.style.opacity = '0';
                setTimeout(() => {
                    spinnerElement.style.display = 'none';
                }, 300);
                csvFileInput.value = ''; // Reset file input
            });
        }
    });

    // Run evaluation button handler with improved animations
    runButton.addEventListener('click', function() {
        const systemPrompt = systemPromptTextarea.value.trim();
        const outputPrompt = outputPromptTextarea.value.trim();
        const examplesText = examplesTextarea.value.trim();
        
        // Validate inputs
        if (!systemPrompt) {
            showAlert('System prompt is required');
            systemPromptTextarea.focus();
            return;
        }
        
        if (!outputPrompt) {
            showAlert('Output prompt is required');
            outputPromptTextarea.focus();
            return;
        }
        
        if (!examplesText) {
            showAlert('Example data is required');
            examplesTextarea.focus();
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
        
        // Show spinner with pulse animation
        spinnerElement.style.opacity = '0';
        spinnerElement.style.display = 'block';
        setTimeout(() => {
            spinnerElement.style.opacity = '1';
        }, 10);
        
        // Disable button with visual feedback
        runButton.disabled = true;
        runButton.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Processing...';
        
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
                // Calculate statistics for performance overview
                let totalScore = 0;
                let perfectCount = 0;
                
                // Display results with staggered animation
                data.results.forEach((result, index) => {
                    const scoreClass = getScoreClass(result.score);
                    const scoreText = getScoreText(result.score);
                    
                    // Update statistics
                    totalScore += result.score;
                    if (result.score >= 0.9) perfectCount++;
                    
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${index + 1}</td>
                        <td><pre>${escapeHtml(result.user_input)}</pre></td>
                        <td><pre>${escapeHtml(result.ground_truth_output)}</pre></td>
                        <td><pre>${escapeHtml(result.model_response)}</pre></td>
                        <td class="${scoreClass}">${scoreText} (${result.score.toFixed(2)})</td>
                    `;
                    
                    // Add staggered animation
                    row.style.opacity = '0';
                    row.style.transform = 'translateY(20px)';
                    row.style.transition = 'all 0.3s ease-out';
                    
                    resultsBody.appendChild(row);
                    
                    // Trigger animation after a delay based on index
                    setTimeout(() => {
                        row.style.opacity = '1';
                        row.style.transform = 'translateY(0)';
                    }, 50 * index);
                });
                
                // Update performance overview
                const avgScore = totalScore / data.results.length;
                avgScoreElement.textContent = avgScore.toFixed(2);
                perfectMatchesElement.textContent = `${perfectCount}/${data.results.length}`;
                totalExamplesElement.textContent = data.results.length;
                
                // Show results section with fade-in
                resultsSection.style.opacity = '0';
                resultsSection.style.display = 'block';
                setTimeout(() => {
                    resultsSection.style.opacity = '1';
                    // Scroll to results with smooth scrolling
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            } else {
                showAlert('No results returned from evaluation');
            }
        })
        .catch(error => {
            showAlert('Error running evaluation: ' + error.message);
        })
        .finally(() => {
            // Hide spinner with fade-out
            spinnerElement.style.opacity = '0';
            setTimeout(() => {
                spinnerElement.style.display = 'none';
            }, 300);
            
            // Reset button
            runButton.disabled = false;
            runButton.innerHTML = '<i class="fa-solid fa-play me-2"></i> Run Evaluation';
        });
    });
    
    // Save prompts button handler with improved feedback
    saveButton.addEventListener('click', function() {
        const systemPrompt = systemPromptTextarea.value.trim();
        const outputPrompt = outputPromptTextarea.value.trim();
        
        // Validate inputs
        if (!systemPrompt) {
            showAlert('System prompt is required');
            systemPromptTextarea.focus();
            return;
        }
        
        if (!outputPrompt) {
            showAlert('Output prompt is required');
            outputPromptTextarea.focus();
            return;
        }
        
        // Prepare data
        const data = {
            system_prompt: systemPrompt,
            output_prompt: outputPrompt
        };
        
        // Show spinner
        spinnerElement.style.opacity = '0';
        spinnerElement.style.display = 'block';
        setTimeout(() => {
            spinnerElement.style.opacity = '1';
        }, 10);
        
        // Disable button with visual feedback
        saveButton.disabled = true;
        saveButton.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Saving...';
        
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
                showAlert(`Prompts saved successfully! <br><strong>System:</strong> ${data.system_filename}<br><strong>Output:</strong> ${data.output_filename}`, 'success');
            }
        })
        .catch(error => {
            showAlert('Error saving prompts: ' + error.message);
        })
        .finally(() => {
            // Hide spinner with fade-out
            spinnerElement.style.opacity = '0';
            setTimeout(() => {
                spinnerElement.style.display = 'none';
            }, 300);
            
            // Reset button
            saveButton.disabled = false;
            saveButton.innerHTML = '<i class="fa-solid fa-save me-2"></i> Save Prompts';
        });
    });
    
    // Enhanced helper functions
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
        if (score > 0.7) return 'Great Match';
        if (score > 0.5) return 'Good Match';
        if (score > 0.3) return 'Fair Match';
        if (score > 0) return 'Partial Match';
        return 'No Match';
    }
    
    // Add keyboard shortcuts for power users
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to run evaluation
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (!runButton.disabled) {
                e.preventDefault();
                runButton.click();
            }
        }
        
        // Ctrl/Cmd + S to save prompts
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            if (!saveButton.disabled) {
                e.preventDefault();
                saveButton.click();
            }
        }
    });
    
    // Add CSS animation for page load
    document.body.classList.add('page-loaded');
});
