document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const experimentSelect = document.getElementById('experiment-select');
    const iterationSelect = document.getElementById('iteration-select');
    const spinnerElement = document.getElementById('spinner');
    const alertContainer = document.getElementById('alert-container');
    
    // Original prompts elements
    const originalSystemPrompt = document.getElementById('original-system-prompt');
    const originalOutputPrompt = document.getElementById('original-output-prompt');
    
    // Evaluator prompts elements
    const evaluatorSystemPrompt = document.getElementById('evaluator-system-prompt');
    const evaluatorOutputPrompt = document.getElementById('evaluator-output-prompt');
    
    // Optimizer prompt element
    const optimizerPrompt = document.getElementById('optimizer-prompt');
    
    // Final prompts elements
    const finalSystemPrompt = document.getElementById('final-system-prompt');
    const finalOutputPrompt = document.getElementById('final-output-prompt');
    
    // Metrics elements
    const improvementPercentage = document.getElementById('improvement-percentage');
    const clarityScore = document.getElementById('clarity-score');
    const concisenessScore = document.getElementById('conciseness-score');
    const effectivenessScore = document.getElementById('effectiveness-score');
    
    // Initialize
    loadExperiments();
    
    // Set up event listeners
    experimentSelect.addEventListener('change', handleExperimentChange);
    iterationSelect.addEventListener('change', handleIterationChange);
    
    // Set up copy buttons
    document.querySelectorAll('.copy-btn').forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                navigator.clipboard.writeText(targetElement.textContent.trim())
                    .then(() => {
                        const originalText = this.innerHTML;
                        this.innerHTML = '<i class="fa-solid fa-check"></i>';
                        
                        setTimeout(() => {
                            this.innerHTML = originalText;
                        }, 2000);
                    })
                    .catch(err => {
                        showAlert('Failed to copy: ' + err, 'danger');
                    });
            }
        });
    });
    
    /**
     * Load all experiments
     */
    function loadExperiments() {
        showSpinner();
        
        fetch('/api/experiments_list')
            .then(response => response.json())
            .then(data => {
                if (data.experiments && data.experiments.length > 0) {
                    // Clear existing options
                    experimentSelect.innerHTML = '<option value="">Select an experiment...</option>';
                    
                    // Sort experiments by date (newest first)
                    const sortedExperiments = [...data.experiments].sort((a, b) => {
                        return new Date(b.created_at) - new Date(a.created_at);
                    });
                    
                    // Add options for each experiment
                    sortedExperiments.forEach(exp => {
                        const option = document.createElement('option');
                        option.value = exp.id;
                        option.textContent = `${exp.id} (${formatDate(exp.created_at)})`;
                        experimentSelect.appendChild(option);
                    });
                    
                    // Select the most recent experiment by default
                    if (sortedExperiments.length > 0) {
                        experimentSelect.value = sortedExperiments[0].id;
                        handleExperimentChange();
                    }
                } else {
                    showAlert('No experiments found', 'warning');
                }
            })
            .catch(error => {
                showAlert('Error loading experiments: ' + error.message, 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    /**
     * Handle experiment selection change
     */
    function handleExperimentChange() {
        const experimentId = experimentSelect.value;
        
        if (!experimentId) {
            iterationSelect.innerHTML = '<option value="">Select an iteration...</option>';
            iterationSelect.disabled = true;
            return;
        }
        
        showSpinner();
        
        fetch(`/api/experiment_data/${experimentId}`)
            .then(response => response.json())
            .then(data => {
                if (data.iterations && data.iterations.length > 0) {
                    // Clear existing options
                    iterationSelect.innerHTML = '<option value="">Select an iteration...</option>';
                    
                    // Add option for the original prompts
                    const originalOption = document.createElement('option');
                    originalOption.value = "original";
                    originalOption.textContent = "Original Prompts";
                    iterationSelect.appendChild(originalOption);
                    
                    // Add options for each iteration
                    data.iterations.forEach((iteration, index) => {
                        const option = document.createElement('option');
                        option.value = index.toString();
                        option.textContent = `Iteration ${index + 1}`;
                        iterationSelect.appendChild(option);
                    });
                    
                    // Enable the iteration select
                    iterationSelect.disabled = false;
                    
                    // Select the original prompts by default
                    iterationSelect.value = "original";
                    
                    // Load original prompts
                    loadPrompts(experimentId, "original");
                } else {
                    showAlert('No iterations found for this experiment', 'warning');
                    iterationSelect.innerHTML = '<option value="">No iterations available</option>';
                    iterationSelect.disabled = true;
                }
            })
            .catch(error => {
                showAlert('Error loading experiment details: ' + error.message, 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    /**
     * Handle iteration selection change
     */
    function handleIterationChange() {
        const experimentId = experimentSelect.value;
        const iterationIndex = iterationSelect.value;
        
        if (!experimentId || !iterationIndex) {
            return;
        }
        
        loadPrompts(experimentId, iterationIndex);
    }
    
    /**
     * Load prompts for the selected experiment and iteration
     */
    function loadPrompts(experimentId, iterationIndex) {
        showSpinner();
        
        fetch(`/api/prompts?experiment_id=${experimentId}&iteration=${iterationIndex}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                    return;
                }
                
                // Update original prompts
                if (data.original) {
                    originalSystemPrompt.textContent = data.original.system_prompt || "No system prompt available";
                    originalOutputPrompt.textContent = data.original.output_prompt || "No output prompt available";
                }
                
                // Update optimizer prompt
                if (data.optimizer) {
                    optimizerPrompt.textContent = data.optimizer.prompt || "No optimizer prompt available";
                }
                
                // Update final prompts (if this is not the original prompts)
                if (iterationIndex !== "original" && data.final) {
                    finalSystemPrompt.textContent = data.final.system_prompt || "No optimized system prompt available";
                    finalOutputPrompt.textContent = data.final.output_prompt || "No optimized output prompt available";
                    
                    // Update metrics if available
                    if (data.metrics) {
                        improvementPercentage.textContent = data.metrics.improvement_percentage ? 
                            `${data.metrics.improvement_percentage}%` : "--";
                        clarityScore.textContent = data.metrics.clarity_score || "--";
                        concisenessScore.textContent = data.metrics.conciseness_score || "--";
                        effectivenessScore.textContent = data.metrics.effectiveness_score || "--";
                    }
                } else {
                    // Reset final prompts if viewing original
                    finalSystemPrompt.textContent = "Select an iteration to view optimized prompts";
                    finalOutputPrompt.textContent = "Select an iteration to view optimized prompts";
                    
                    // Reset metrics
                    improvementPercentage.textContent = "--";
                    clarityScore.textContent = "--";
                    concisenessScore.textContent = "--";
                    effectivenessScore.textContent = "--";
                }
            })
            .catch(error => {
                showAlert('Error loading prompts: ' + error.message, 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    /**
     * Format date to a readable string
     */
    function formatDate(dateString) {
        const date = new Date(dateString);
        const formatter = new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        return formatter.format(date);
    }
    
    /**
     * Show the spinner
     */
    function showSpinner() {
        spinnerElement.style.display = 'block';
    }
    
    /**
     * Hide the spinner
     */
    function hideSpinner() {
        spinnerElement.style.display = 'none';
    }
    
    /**
     * Show an alert message
     */
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
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alertElement);
            bsAlert.close();
        }, 5000);
    }
});