/**
 * Final Prompts Page JavaScript
 * This script handles the final prompts interface with testing and export functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // DOM elements
    const systemPromptTextarea = document.getElementById('system-prompt');
    const outputPromptTextarea = document.getElementById('output-prompt');
    const experimentSelect = document.getElementById('experiment-select');
    const versionTimeline = document.getElementById('version-timeline');
    const exportJsonBtn = document.getElementById('export-json-btn');
    const exportTxtBtn = document.getElementById('export-txt-btn');
    const copySystemBtn = document.getElementById('copy-system-btn');
    const copyOutputBtn = document.getElementById('copy-output-btn');
    const saveChangesBtn = document.getElementById('save-changes-btn');
    const testInput = document.getElementById('test-input');
    const runTestBtn = document.getElementById('run-test-btn');
    const testResultsContainer = document.getElementById('test-results-container');
    const resultInput = document.getElementById('result-input');
    const resultResponse = document.getElementById('result-response');
    const resultTime = document.getElementById('result-time');
    const resultQuality = document.getElementById('result-quality');
    const getGraderFeedbackBtn = document.getElementById('get-grader-feedback-btn');
    const graderFeedback = document.getElementById('grader-feedback');
    const saveFinalBtn = document.getElementById('save-final-btn');
    const continueOptimizationBtn = document.getElementById('continue-optimization-btn');
    const spinner = document.getElementById('spinner');
    const alertContainer = document.getElementById('alert-container');
    
    // Performance metrics elements
    const trainingScoreEl = document.getElementById('training-score');
    const validationScoreEl = document.getElementById('validation-score');
    const perfectMatchesEl = document.getElementById('perfect-matches');
    const improvementEl = document.getElementById('improvement');
    
    // Chart
    let performanceChart = null;
    
    // Current experiment data
    let currentExperimentId = null;
    let currentIteration = null;
    
    // Check for URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const experimentParam = urlParams.get('experiment');
    const iterationParam = urlParams.get('iteration');
    
    // Check for temporarily stored prompts
    const tempSystemPrompt = localStorage.getItem('temp_system_prompt');
    const tempOutputPrompt = localStorage.getItem('temp_output_prompt');
    
    // Initialize
    initializePage();
    
    // Event listeners
    experimentSelect.addEventListener('change', loadExperimentVersions);
    exportJsonBtn.addEventListener('click', exportAsJson);
    exportTxtBtn.addEventListener('click', exportAsTxt);
    copySystemBtn.addEventListener('click', copySystemPrompt);
    copyOutputBtn.addEventListener('click', copyOutputPrompt);
    saveChangesBtn.addEventListener('click', saveChanges);
    runTestBtn.addEventListener('click', runTest);
    getGraderFeedbackBtn.addEventListener('click', getGraderFeedback);
    saveFinalBtn.addEventListener('click', saveFinalVersion);
    continueOptimizationBtn.addEventListener('click', continueOptimization);
    
    /**
     * Initialize the page
     */
    function initializePage() {
        // Load experiments
        loadExperiments();
        
        // If URL parameters are present, load that experiment and iteration
        if (experimentParam && iterationParam) {
            currentExperimentId = experimentParam;
            currentIteration = iterationParam;
            
            // Once experiments are loaded, select the right one
            setTimeout(() => {
                if (experimentSelect.querySelector(`option[value="${experimentParam}"]`)) {
                    experimentSelect.value = experimentParam;
                    loadExperimentVersions();
                }
            }, 1000);
        }
        
        // If temp prompts exist, load them
        if (tempSystemPrompt && tempOutputPrompt) {
            systemPromptTextarea.value = tempSystemPrompt;
            outputPromptTextarea.value = tempOutputPrompt;
            
            // Clear localStorage
            localStorage.removeItem('temp_system_prompt');
            localStorage.removeItem('temp_output_prompt');
        }
        
        // Initialize performance chart
        initPerformanceChart();
    }
    
    /**
     * Load available experiments for selection
     */
    function loadExperiments() {
        showSpinner();
        
        fetch('/experiments')
            .then(response => response.json())
            .then(data => {
                hideSpinner();
                
                if (data.experiments && data.experiments.length > 0) {
                    experimentSelect.innerHTML = '<option value="" selected>Select an experiment...</option>';
                    
                    data.experiments.forEach(experiment => {
                        const option = document.createElement('option');
                        option.value = experiment.id;
                        option.textContent = `${experiment.id} (${new Date(experiment.created_at).toLocaleString()})`;
                        experimentSelect.appendChild(option);
                    });
                } else {
                    showAlert('No experiments found. Please run training first.', 'warning');
                }
            })
            .catch(error => {
                hideSpinner();
                showAlert(`Error loading experiments: ${error}`, 'danger');
                console.error('Error loading experiments:', error);
            });
    }
    
    /**
     * Load versions/iterations for the selected experiment
     */
    function loadExperimentVersions() {
        const experimentId = experimentSelect.value;
        currentExperimentId = experimentId;
        
        if (!experimentId) {
            versionTimeline.innerHTML = '<li>No experiment selected</li>';
            return;
        }
        
        showSpinner();
        
        fetch(`/experiments/${experimentId}`)
            .then(response => response.json())
            .then(data => {
                hideSpinner();
                
                if (data.iterations && data.iterations.length > 0) {
                    versionTimeline.innerHTML = '';
                    
                    // Sort iterations by number
                    const sortedIterations = [...data.iterations].sort((a, b) => b.iteration - a.iteration);
                    
                    // Add initial version (iteration 0)
                    if (!sortedIterations.find(i => i.iteration === 0)) {
                        sortedIterations.push({
                            iteration: 0,
                            avg_score: data.initial_score || 0,
                            created_at: data.created_at
                        });
                    }
                    
                    // Add each iteration to the timeline
                    sortedIterations.forEach(iteration => {
                        const isActive = iteration.iteration === parseInt(currentIteration);
                        const li = document.createElement('li');
                        li.className = isActive ? 'active' : '';
                        li.dataset.iteration = iteration.iteration;
                        
                        // Format date
                        const dateStr = iteration.created_at 
                            ? new Date(iteration.created_at).toLocaleString()
                            : 'N/A';
                            
                        const iterationName = iteration.iteration === 0 
                            ? 'Initial Version' 
                            : `Iteration ${iteration.iteration}`;
                            
                        li.innerHTML = `
                            <div class="version-title">${iterationName}${isActive ? ' (Current)' : ''}</div>
                            <div class="version-date">${dateStr}</div>
                            <div class="version-score">Score: ${iteration.avg_score.toFixed(2)}</div>
                        `;
                        
                        // Add click event to load this version
                        li.addEventListener('click', () => loadPromptVersion(experimentId, iteration.iteration));
                        
                        versionTimeline.appendChild(li);
                    });
                    
                    // If currentIteration is set, load that version
                    if (currentIteration !== null) {
                        loadPromptVersion(experimentId, currentIteration);
                    } else if (sortedIterations.length > 0) {
                        // Otherwise load the latest iteration
                        const latestIteration = Math.max(...sortedIterations.map(i => i.iteration));
                        loadPromptVersion(experimentId, latestIteration);
                    }
                    
                    // Load performance data
                    loadPerformanceData(experimentId);
                } else {
                    versionTimeline.innerHTML = '<li>No iterations found for this experiment</li>';
                    showAlert('No iterations found for this experiment.', 'warning');
                }
            })
            .catch(error => {
                hideSpinner();
                showAlert(`Error loading experiment versions: ${error}`, 'danger');
                console.error('Error loading experiment versions:', error);
            });
    }
    
    /**
     * Load a specific prompt version
     */
    function loadPromptVersion(experimentId, iteration) {
        currentExperimentId = experimentId;
        currentIteration = iteration;
        
        showSpinner();
        
        fetch(`/experiments/${experimentId}/iterations/${iteration}/examples`)
            .then(response => response.json())
            .then(data => {
                hideSpinner();
                
                if (data.system_prompt && data.output_prompt) {
                    systemPromptTextarea.value = data.system_prompt;
                    outputPromptTextarea.value = data.output_prompt;
                    
                    // Update active class in timeline
                    const items = versionTimeline.querySelectorAll('li');
                    items.forEach(item => {
                        if (parseInt(item.dataset.iteration) === parseInt(iteration)) {
                            item.classList.add('active');
                        } else {
                            item.classList.remove('active');
                        }
                    });
                    
                    showAlert(`Loaded version ${iteration} of experiment ${experimentId}`, 'success');
                } else {
                    showAlert('No prompts found for this iteration.', 'warning');
                }
            })
            .catch(error => {
                hideSpinner();
                showAlert(`Error loading prompt version: ${error}`, 'danger');
                console.error('Error loading prompt version:', error);
            });
    }
    
    /**
     * Load performance data for the experiment
     */
    function loadPerformanceData(experimentId) {
        showSpinner();
        
        fetch(`/experiments/${experimentId}`)
            .then(response => response.json())
            .then(data => {
                hideSpinner();
                
                if (data.iterations && data.iterations.length > 0) {
                    // Extract scores for performance chart
                    const scores = data.iterations.map(iteration => ({
                        iteration: iteration.iteration,
                        score: iteration.avg_score,
                        validation_score: iteration.validation_score || 0
                    }));
                    
                    // Sort by iteration
                    scores.sort((a, b) => a.iteration - b.iteration);
                    
                    // Get the latest scores
                    const latestIteration = scores[scores.length - 1];
                    const initialIteration = scores[0] || { score: 0, validation_score: 0 };
                    
                    // Calculate improvement
                    const improvement = initialIteration.score > 0 
                        ? ((latestIteration.score - initialIteration.score) / initialIteration.score * 100).toFixed(1) 
                        : 'N/A';
                    
                    // Update UI
                    trainingScoreEl.textContent = latestIteration.score.toFixed(2);
                    validationScoreEl.textContent = latestIteration.validation_score.toFixed(2);
                    
                    // Calculate perfect matches (if available)
                    const perfectMatches = data.perfect_match_percent || 'N/A';
                    perfectMatchesEl.textContent = typeof perfectMatches === 'number' 
                        ? `${perfectMatches.toFixed(1)}%` 
                        : perfectMatches;
                    
                    improvementEl.textContent = improvement === 'N/A' ? improvement : `+${improvement}%`;
                    
                    // Update chart
                    updatePerformanceChart(scores);
                }
            })
            .catch(error => {
                hideSpinner();
                console.error('Error loading performance data:', error);
            });
    }
    
    /**
     * Initialize the performance chart
     */
    function initPerformanceChart() {
        const ctx = document.getElementById('performance-chart').getContext('2d');
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Initial', 'Iteration 1', 'Iteration 2', 'Iteration 3'],
                datasets: [
                    {
                        label: 'Training Score',
                        data: [0.65, 0.78, 0.85, 0.92],
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        tension: 0.1
                    },
                    {
                        label: 'Validation Score',
                        data: [0.60, 0.73, 0.81, 0.89],
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.5,
                        max: 1.0
                    }
                }
            }
        });
    }
    
    /**
     * Update the performance chart with real data
     */
    function updatePerformanceChart(scores) {
        if (performanceChart) {
            performanceChart.destroy();
        }
        
        const labels = scores.map(s => s.iteration === 0 ? 'Initial' : `Iteration ${s.iteration}`);
        const trainingData = scores.map(s => s.score);
        const validationData = scores.map(s => s.validation_score || null);
        
        const ctx = document.getElementById('performance-chart').getContext('2d');
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Training Score',
                        data: trainingData,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        tension: 0.1
                    },
                    {
                        label: 'Validation Score',
                        data: validationData,
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: Math.max(0, Math.min(...trainingData, ...validationData.filter(v => v !== null)) - 0.1),
                        max: 1.0
                    }
                }
            }
        });
    }
    
    /**
     * Export prompts as JSON
     */
    function exportAsJson() {
        const jsonData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value,
            experiment_id: currentExperimentId,
            iteration: currentIteration,
            exported_at: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `prompts_${currentExperimentId || 'custom'}_${currentIteration || 'custom'}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Export prompts as TXT
     */
    function exportAsTxt() {
        const txtContent = 
            `SYSTEM PROMPT:\n${systemPromptTextarea.value}\n\n` +
            `OUTPUT PROMPT:\n${outputPromptTextarea.value}\n\n` +
            `Exported from experiment: ${currentExperimentId || 'Custom'}\n` +
            `Iteration: ${currentIteration || 'Custom'}\n` +
            `Date: ${new Date().toLocaleString()}`;
            
        const blob = new Blob([txtContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `prompts_${currentExperimentId || 'custom'}_${currentIteration || 'custom'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Copy system prompt to clipboard
     */
    function copySystemPrompt() {
        navigator.clipboard.writeText(systemPromptTextarea.value)
            .then(() => {
                showAlert('System prompt copied to clipboard', 'success');
            })
            .catch(err => {
                showAlert('Failed to copy: ' + err, 'danger');
            });
    }
    
    /**
     * Copy output prompt to clipboard
     */
    function copyOutputPrompt() {
        navigator.clipboard.writeText(outputPromptTextarea.value)
            .then(() => {
                showAlert('Output prompt copied to clipboard', 'success');
            })
            .catch(err => {
                showAlert('Failed to copy: ' + err, 'danger');
            });
    }
    
    /**
     * Save changes to the prompts
     */
    function saveChanges() {
        if (!systemPromptTextarea.value || !outputPromptTextarea.value) {
            showAlert('Both system prompt and output prompt are required', 'warning');
            return;
        }
        
        const saveData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value,
            experiment_id: currentExperimentId,
            iteration: currentIteration
        };
        
        showSpinner();
        
        fetch('/save_prompts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(saveData)
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }
            
            showAlert('Prompts saved successfully', 'success');
            
            // Update current experiment and iteration if they were created
            if (data.experiment_id) {
                currentExperimentId = data.experiment_id;
            }
            
            if (data.iteration !== undefined) {
                currentIteration = data.iteration;
            }
        })
        .catch(error => {
            hideSpinner();
            showAlert(`Error saving prompts: ${error}`, 'danger');
            console.error('Error saving prompts:', error);
        });
    }
    
    /**
     * Run a test with the current prompts
     */
    function runTest() {
        if (!testInput.value.trim()) {
            showAlert('Please enter test input', 'warning');
            return;
        }
        
        if (!systemPromptTextarea.value || !outputPromptTextarea.value) {
            showAlert('Both system prompt and output prompt are required', 'warning');
            return;
        }
        
        const testData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value,
            user_input: testInput.value
        };
        
        showSpinner();
        
        // This endpoint needs to be implemented in the backend
        fetch('/test_prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(testData)
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }
            
            // Show the test results
            testResultsContainer.style.display = 'block';
            resultInput.textContent = testInput.value.substring(0, 100) + (testInput.value.length > 100 ? '...' : '');
            resultResponse.textContent = data.response || 'No response received';
            resultTime.textContent = `Response time: ${data.response_time || 'N/A'}`;
            
            // Set quality badge
            const quality = data.quality || 'Unknown';
            resultQuality.textContent = quality;
            
            // Set badge color based on quality
            resultQuality.className = 'badge ';
            switch (quality.toLowerCase()) {
                case 'high quality':
                    resultQuality.className += 'bg-success';
                    break;
                case 'good':
                    resultQuality.className += 'bg-primary';
                    break;
                case 'average':
                    resultQuality.className += 'bg-info';
                    break;
                case 'low quality':
                    resultQuality.className += 'bg-warning';
                    break;
                default:
                    resultQuality.className += 'bg-secondary';
            }
        })
        .catch(error => {
            hideSpinner();
            showAlert(`Error running test: ${error}`, 'danger');
            console.error('Error running test:', error);
        });
    }
    
    /**
     * Get grader feedback on the current prompts
     */
    function getGraderFeedback() {
        if (!systemPromptTextarea.value || !outputPromptTextarea.value) {
            showAlert('Both system prompt and output prompt are required', 'warning');
            return;
        }
        
        const gradingData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value
        };
        
        showSpinner();
        
        // This endpoint needs to be implemented in the backend
        fetch('/grade_prompts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(gradingData)
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }
            
            // Display grader feedback
            let feedbackHtml = '';
            
            if (data.feedback) {
                if (typeof data.feedback === 'object') {
                    // If it's a structured object
                    const feedback = data.feedback;
                    feedbackHtml = `
                        <p class="mb-2"><strong>Clarity:</strong> ${feedback.clarity}/10 - ${feedback.clarity_comment || ''}</p>
                        <p class="mb-2"><strong>Conciseness:</strong> ${feedback.conciseness}/10 - ${feedback.conciseness_comment || ''}</p>
                        <p class="mb-2"><strong>Effectiveness:</strong> ${feedback.effectiveness}/10 - ${feedback.effectiveness_comment || ''}</p>
                        <p class="mb-0"><strong>Overall Score:</strong> ${feedback.overall_score}/1.0</p>
                    `;
                } else if (typeof data.feedback === 'string') {
                    // If it's just a string
                    feedbackHtml = `<p>${data.feedback}</p>`;
                }
            } else {
                feedbackHtml = '<p>No detailed feedback available.</p>';
            }
            
            graderFeedback.innerHTML = feedbackHtml;
        })
        .catch(error => {
            hideSpinner();
            showAlert(`Error getting grader feedback: ${error}`, 'danger');
            console.error('Error getting grader feedback:', error);
        });
    }
    
    /**
     * Save as final version
     */
    function saveFinalVersion() {
        if (!systemPromptTextarea.value || !outputPromptTextarea.value) {
            showAlert('Both system prompt and output prompt are required', 'warning');
            return;
        }
        
        const finalData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value,
            experiment_id: currentExperimentId,
            iteration: currentIteration,
            is_final: true
        };
        
        showSpinner();
        
        fetch('/save_prompts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(finalData)
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }
            
            showAlert('Prompts saved as final version!', 'success');
            
            // Update current experiment and iteration if they were created
            if (data.experiment_id) {
                currentExperimentId = data.experiment_id;
            }
            
            if (data.iteration !== undefined) {
                currentIteration = data.iteration;
            }
        })
        .catch(error => {
            hideSpinner();
            showAlert(`Error saving final version: ${error}`, 'danger');
            console.error('Error saving final version:', error);
        });
    }
    
    /**
     * Continue optimization process
     */
    function continueOptimization() {
        if (!systemPromptTextarea.value || !outputPromptTextarea.value) {
            showAlert('Both system prompt and output prompt are required', 'warning');
            return;
        }
        
        // Save current prompts to local storage
        localStorage.setItem('temp_system_prompt', systemPromptTextarea.value);
        localStorage.setItem('temp_output_prompt', outputPromptTextarea.value);
        
        // Redirect to training page
        window.location.href = `/training?experiment=${currentExperimentId || ''}`;
    }
    
    /**
     * Show a bootstrap alert
     */
    function showAlert(message, type) {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        alertContainer.innerHTML = '';
        alertContainer.appendChild(alert);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    }
    
    /**
     * Show the loading spinner
     */
    function showSpinner() {
        spinner.style.display = 'block';
    }
    
    /**
     * Hide the loading spinner
     */
    function hideSpinner() {
        spinner.style.display = 'none';
    }
});