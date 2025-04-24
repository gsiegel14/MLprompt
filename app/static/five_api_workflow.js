/**
 * ATLAS Autonomous Prompt Workflow Module
 * Handles the 5-step prompt optimization process using Vertex AI and Hugging Face
 */

// Workflow state tracking
const workflowState = {
    inProgress: false,
    currentStep: 0,
    examples: [],
    metricsChart: null,
    validationSplit: 0.8 // Default training/validation split ratio
};

document.addEventListener('DOMContentLoaded', function() {
    initializeUI();
    setupEventListeners();
});

/**
 * Initialize UI components
 */
function initializeUI() {
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        new bootstrap.Tooltip(tooltip);
    });
    
    // Set up example UI
    const addExampleBtn = document.getElementById('add-example-btn');
    const loadFromCsvBtn = document.getElementById('load-from-csv-btn');
    
    if (addExampleBtn) {
        addExampleBtn.addEventListener('click', function() {
            const exampleModal = new bootstrap.Modal(document.getElementById('exampleModal'));
            exampleModal.show();
        });
    }
    
    if (loadFromCsvBtn) {
        loadFromCsvBtn.addEventListener('click', function() {
            const csvModal = new bootstrap.Modal(document.getElementById('csvModal'));
            csvModal.show();
        });
    }
    
    // Initialize copy buttons
    initializeCopyButtons();
    
    // Initialize data stats
    updateDataStats();
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Optimization start button
    const startOptimizationBtn = document.getElementById('start-optimization');
    if (startOptimizationBtn) {
        startOptimizationBtn.addEventListener('click', startOptimization);
    }
    
    // Save example button
    const saveExampleBtn = document.getElementById('save-example-btn');
    if (saveExampleBtn) {
        saveExampleBtn.addEventListener('click', saveExample);
    }
    
    // Load CSV button
    const loadCsvBtn = document.getElementById('load-csv-btn');
    if (loadCsvBtn) {
        loadCsvBtn.addEventListener('click', loadCsvExamples);
    }
    
    // Use optimized prompts button
    const useOptimizedPromptsBtn = document.getElementById('use-optimized-prompts');
    if (useOptimizedPromptsBtn) {
        useOptimizedPromptsBtn.addEventListener('click', useOptimizedPrompts);
    }
    
    // NEJM Dataset buttons
    const loadNejmTrainBtn = document.getElementById('load-nejm-train-btn');
    if (loadNejmTrainBtn) {
        loadNejmTrainBtn.addEventListener('click', () => loadNejmDataset('train'));
    }
    
    const loadNejmValidationBtn = document.getElementById('load-nejm-validation-btn');
    if (loadNejmValidationBtn) {
        loadNejmValidationBtn.addEventListener('click', () => loadNejmDataset('validation'));
    }
    
    const resetNejmCacheBtn = document.getElementById('reset-nejm-cache-btn');
    if (resetNejmCacheBtn) {
        resetNejmCacheBtn.addEventListener('click', resetNejmCache);
    }
}

/**
 * Start the optimization process
 */
function startOptimization() {
    // Prevent multiple runs
    if (workflowState.inProgress) {
        return;
    }
    
    // Validate inputs
    const systemPrompt = document.getElementById('system-prompt').value.trim();
    const outputPrompt = document.getElementById('output-prompt').value.trim();
    
    if (!systemPrompt || !outputPrompt) {
        showAlert('Please provide both system and output prompts.', 'danger');
        return;
    }
    
    if (workflowState.examples.length === 0) {
        showAlert('Please add at least one example for optimization.', 'danger');
        return;
    }
    
    // Get settings
    const primaryModel = document.getElementById('primary-model').value;
    const optimizerModel = document.getElementById('optimizer-model').value;
    const maxIterations = parseInt(document.getElementById('max-iterations').value) || 3;
    const strategy = document.getElementById('optimization-strategy').value;
    
    // Update state
    workflowState.inProgress = true;
    workflowState.currentStep = 1;
    
    // Update UI
    updateStepIndicators();
    document.getElementById('optimization-progress').classList.remove('d-none');
    document.getElementById('progress-bar').style.width = '20%';
    document.getElementById('progress-bar').textContent = '20%';
    document.getElementById('progress-status').textContent = 'Step 1: Running primary LLM...';
    document.getElementById('start-optimization').disabled = true;
    
    // Prepare request data
    const requestData = {
        system_prompt: systemPrompt,
        output_prompt: outputPrompt,
        examples: workflowState.examples,
        primary_model: primaryModel,
        optimizer_model: optimizerModel,
        max_iterations: maxIterations,
        strategy: strategy
    };
    
    // Make API request
    fetch('/api/five_api_workflow', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        processWorkflowResults(data);
    })
    .catch(error => {
        console.error('Error in optimization process:', error);
        showAlert('Error occurred during optimization: ' + error.message, 'danger');
        resetWorkflowState();
    });
    
    // Simulate progress updates for demo purposes
    simulateWorkflowProgress();
}

/**
 * Process workflow results from the API
 */
function processWorkflowResults(data) {
    // Update state
    workflowState.inProgress = false;
    workflowState.currentStep = 5; // Completed all steps
    
    // Update UI
    updateStepIndicators();
    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('progress-bar').textContent = '100%';
    document.getElementById('progress-status').textContent = 'Optimization complete!';
    document.getElementById('start-optimization').disabled = false;
    
    // Display results
    document.getElementById('results-container').classList.remove('d-none');
    
    // Set optimized prompts
    if (data.optimized_prompts) {
        document.getElementById('best-system-prompt').value = data.optimized_prompts.system_prompt;
        document.getElementById('best-output-prompt').value = data.optimized_prompts.output_prompt;
    }
    
    // Update metrics
    updateMetricsDisplay(data.metrics);
    
    // Show success message
    showAlert('Prompt optimization completed successfully!', 'success');
}

/**
 * Update the metrics display with data from the API
 */
function updateMetricsDisplay(metrics) {
    if (!metrics) return;
    
    // Update metrics table
    const metricsTableBody = document.getElementById('metrics-table-body');
    if (metricsTableBody) {
        metricsTableBody.innerHTML = '';
        
        // Add rows for each metric
        const metricNames = {
            'exact_match': 'Exact Match',
            'semantic_similarity': 'Semantic Similarity',
            'keyword_match': 'Keyword Match',
            'bleu': 'BLEU Score',
            'rouge': 'ROUGE Score'
        };
        
        for (const [key, label] of Object.entries(metricNames)) {
            if (metrics[key]) {
                const row = document.createElement('tr');
                
                row.innerHTML = `
                    <td>${label}</td>
                    <td>${formatPercent(metrics[key].original?.training || 0)}</td>
                    <td>${formatPercent(metrics[key].original?.validation || 0)}</td>
                    <td>${formatPercent(metrics[key].optimized?.training || 0)}</td>
                    <td>${formatPercent(metrics[key].optimized?.validation || 0)}</td>
                `;
                
                metricsTableBody.appendChild(row);
            }
        }
    }
    
    // Create or update chart
    createMetricsChart(metrics);
}

/**
 * Create a chart to visualize metrics
 */
function createMetricsChart(metrics) {
    const chartCanvas = document.getElementById('metrics-chart');
    if (!chartCanvas) return;
    
    // Destroy existing chart if it exists
    if (workflowState.metricsChart) {
        workflowState.metricsChart.destroy();
    }
    
    // Prepare data for the chart
    const labels = [];
    const originalTraining = [];
    const originalValidation = [];
    const optimizedTraining = [];
    const optimizedValidation = [];
    
    // Format data for charting
    for (const [key, value] of Object.entries(metrics)) {
        if (key !== 'total_examples' && typeof value.original?.training === 'number') {
            const label = key.replace(/_/g, ' ');
            labels.push(label);
            originalTraining.push((value.original?.training || 0) * 100);
            originalValidation.push((value.original?.validation || 0) * 100);
            optimizedTraining.push((value.optimized?.training || 0) * 100);
            optimizedValidation.push((value.optimized?.validation || 0) * 100);
        }
    }
    
    // Create chart
    workflowState.metricsChart = new Chart(chartCanvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Original (Training)',
                    data: originalTraining,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Original (Validation)',
                    data: originalValidation,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Optimized (Training)',
                    data: optimizedTraining,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Optimized (Validation)',
                    data: optimizedValidation,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Score (%)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Metrics Comparison'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update step indicators based on current progress
 */
function updateStepIndicators() {
    // Reset all steps
    for (let i = 1; i <= 5; i++) {
        const stepElement = document.getElementById(`step${i}-indicator`);
        if (stepElement) {
            stepElement.classList.remove('active', 'completed');
        }
    }
    
    // Update based on current step
    for (let i = 1; i <= 5; i++) {
        const stepElement = document.getElementById(`step${i}-indicator`);
        if (stepElement) {
            if (i < workflowState.currentStep) {
                stepElement.classList.add('completed');
            } else if (i === workflowState.currentStep) {
                stepElement.classList.add('active');
            }
        }
    }
}

/**
 * Simulate workflow progress (for demo purposes)
 */
function simulateWorkflowProgress() {
    if (!workflowState.inProgress) return;
    
    // Simulate step progression (only for demonstration)
    const progressInterval = setInterval(() => {
        if (!workflowState.inProgress) {
            clearInterval(progressInterval);
            return;
        }
        
        // Move to next step after delay
        if (workflowState.currentStep < 5) {
            workflowState.currentStep++;
            updateStepIndicators();
            
            // Update progress bar
            const progressPercent = workflowState.currentStep * 20;
            document.getElementById('progress-bar').style.width = `${progressPercent}%`;
            document.getElementById('progress-bar').textContent = `${progressPercent}%`;
            
            // Update status text
            const statusText = {
                1: 'Step 1: Running primary LLM...',
                2: 'Step 2: Evaluating baseline metrics...',
                3: 'Step 3: Optimizing prompts...',
                4: 'Step 4: Testing optimized prompts...',
                5: 'Step 5: Final evaluation...'
            };
            
            document.getElementById('progress-status').textContent = statusText[workflowState.currentStep];
        } else {
            clearInterval(progressInterval);
        }
    }, 3000);
}

/**
 * Save an example from the modal
 */
function saveExample() {
    const userInput = document.getElementById('example-input').value.trim();
    const expectedOutput = document.getElementById('example-output').value.trim();
    
    if (!userInput || !expectedOutput) {
        showAlert('Please provide both input and expected output.', 'warning');
        return;
    }
    
    // Add to examples array
    workflowState.examples.push({
        user_input: userInput,
        ground_truth_output: expectedOutput
    });
    
    // Update UI
    updateExamplesUI();
    
    // Close modal
    bootstrap.Modal.getInstance(document.getElementById('exampleModal')).hide();
    
    // Clear form
    document.getElementById('example-input').value = '';
    document.getElementById('example-output').value = '';
    
    // Show success message
    showAlert('Example added successfully.', 'success');
}

/**
 * Load examples from CSV
 */
function loadCsvExamples() {
    const fileInput = document.getElementById('csv-file');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showAlert('Please select a CSV file.', 'warning');
        return;
    }
    
    const file = fileInput.files[0];
    const reader = new FileReader();
    
    reader.onload = function(e) {
        try {
            const content = e.target.result;
            const rows = content.split('\\n');
            
            // Check for header row
            const header = rows[0].toLowerCase();
            const hasHeader = header.includes('user_input') || 
                              header.includes('input') || 
                              header.includes('ground_truth');
            
            // Parse CSV (simple implementation)
            const examples = [];
            const startIndex = hasHeader ? 1 : 0;
            
            for (let i = startIndex; i < rows.length; i++) {
                const row = rows[i].trim();
                if (!row) continue;
                
                const columns = row.split(',');
                if (columns.length < 2) continue;
                
                examples.push({
                    user_input: columns[0].trim(),
                    ground_truth_output: columns[1].trim()
                });
            }
            
            // Update state
            workflowState.examples = workflowState.examples.concat(examples);
            updateExamplesUI();
            
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById('csvModal')).hide();
            
            // Show success message
            showAlert(`Successfully loaded ${examples.length} examples.`, 'success');
            
        } catch (error) {
            console.error('Error parsing CSV:', error);
            showAlert('Failed to parse CSV: ' + error.message, 'danger');
        }
    };
    
    reader.readAsText(file);
}

/**
 * Update the examples UI
 */
function updateExamplesUI() {
    const container = document.getElementById('examples-container');
    const noExamplesMessage = document.getElementById('no-examples-message');
    
    if (workflowState.examples.length === 0) {
        if (noExamplesMessage) noExamplesMessage.style.display = 'block';
        return;
    }
    
    if (noExamplesMessage) noExamplesMessage.style.display = 'none';
    
    // Create examples list
    let html = '<div class="examples-list">';
    
    workflowState.examples.forEach((example, index) => {
        html += `
            <div class="example-item card mb-2">
                <div class="card-header p-2 d-flex justify-content-between align-items-center">
                    <span class="small">Example #${index + 1}</span>
                    <button class="btn btn-sm btn-outline-danger remove-example" data-index="${index}">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="card-body p-2">
                    <div class="small text-muted mb-1">Input:</div>
                    <div class="mb-2 small">${truncateText(example.user_input, 100)}</div>
                    <div class="small text-muted mb-1">Expected Output:</div>
                    <div class="small">${truncateText(example.ground_truth_output, 100)}</div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
    
    // Add event listeners for remove buttons
    document.querySelectorAll('.remove-example').forEach(button => {
        button.addEventListener('click', function() {
            const index = parseInt(this.getAttribute('data-index'));
            workflowState.examples.splice(index, 1);
            updateExamplesUI();
        });
    });
}

/**
 * Use the optimized prompts
 */
function useOptimizedPrompts() {
    const bestSystemPrompt = document.getElementById('best-system-prompt').value;
    const bestOutputPrompt = document.getElementById('best-output-prompt').value;
    
    if (bestSystemPrompt && bestOutputPrompt) {
        document.getElementById('system-prompt').value = bestSystemPrompt;
        document.getElementById('output-prompt').value = bestOutputPrompt;
        showAlert('Optimized prompts have been applied.', 'success');
    } else {
        showAlert('No optimized prompts available.', 'warning');
    }
}

/**
 * Initialize copy buttons
 */
function initializeCopyButtons() {
    document.querySelectorAll('.copy-button').forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.select();
                document.execCommand('copy');
                
                // Show temporary feedback
                const originalHTML = this.innerHTML;
                this.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    this.innerHTML = originalHTML;
                }, 1500);
            }
        });
    });
}

/**
 * Reset the workflow state
 */
function resetWorkflowState() {
    workflowState.inProgress = false;
    workflowState.currentStep = 0;
    
    // Update UI
    updateStepIndicators();
    document.getElementById('optimization-progress').classList.add('d-none');
    document.getElementById('start-optimization').disabled = false;
}

/**
 * Show an alert message
 */
function showAlert(message, type) {
    const alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 150);
    }, 5000);
}

/**
 * Load NEJM dataset (train or validation)
 * @param {string} datasetType - 'train' or 'validation'
 */
function loadNejmDataset(datasetType) {
    showAlert(`Loading NEJM ${datasetType} dataset...`, 'info');
    
    fetch(`/load_dataset_api?type=nejm_${datasetType}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else if (data.examples && data.examples.length > 0) {
                // Store the examples in our state
                workflowState.examples = data.examples;
                
                // Update UI
                updateExamplesUI();
                updateDataStats();
                
                // Also load specialized medical prompts if not already set
                if (!document.getElementById('system-prompt').value.trim()) {
                    fetch('/load_dataset_api?type=nejm_prompts')
                        .then(response => response.json())
                        .then(promptData => {
                            if (promptData.system_prompt) {
                                document.getElementById('system-prompt').value = promptData.system_prompt;
                                document.getElementById('output-prompt').value = promptData.output_prompt || '';
                                showAlert('Loaded NEJM specialized prompts', 'success');
                            }
                        })
                        .catch(error => console.error('Error loading NEJM prompts:', error));
                }
                
                showAlert(`Loaded ${data.examples.length} NEJM ${datasetType} examples`, 'success');
            } else {
                showAlert(`No NEJM ${datasetType} examples found`, 'warning');
            }
        })
        .catch(error => {
            console.error(`Error loading NEJM ${datasetType} dataset:`, error);
            showAlert(`Error loading NEJM ${datasetType} dataset`, 'danger');
        });
}

/**
 * Reset NEJM data cache and regenerate datasets
 */
function resetNejmCache() {
    showAlert('Resetting NEJM data cache...', 'info');
    
    fetch('/reset_nejm_cache_api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            // Run the fix_nejm_data.py script to regenerate the datasets
            showAlert('Regenerating NEJM datasets...', 'info');
            return fetch('/regenerate_nejm_data_api', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
        }
    })
    .then(response => {
        if (response && !response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        if (response) return response.json();
    })
    .then(data => {
        if (data && data.error) {
            showAlert(data.error, 'danger');
        } else if (data) {
            showAlert(data.message + '. Try loading the NEJM datasets again.', 'success');
        }
    })
    .catch(error => {
        console.error('Error resetting NEJM cache:', error);
        showAlert('Error resetting NEJM cache', 'danger');
    });
}

/**
 * Update data statistics display based on current examples
 */
function updateDataStats() {
    const totalCount = workflowState.examples.length;
    const trainCount = Math.floor(totalCount * workflowState.validationSplit);
    const valCount = totalCount - trainCount;
    
    const statsHTML = `
        <div class="row text-center">
            <div class="col">
                <div class="fs-5">${totalCount}</div>
                <div class="small text-muted">Examples</div>
            </div>
            <div class="col">
                <div class="fs-5">${trainCount}</div>
                <div class="small text-muted">Train</div>
            </div>
            <div class="col">
                <div class="fs-5">${valCount}</div>
                <div class="small text-muted">Validation</div>
            </div>
        </div>
    `;
    
    const dataStatsEl = document.getElementById('data-stats');
    if (dataStatsEl) {
        dataStatsEl.innerHTML = statsHTML;
    }
}

/**
 * Format a number as a percentage
 */
function formatPercent(value) {
    return (value * 100).toFixed(1) + '%';
}

/**
 * Truncate text to a specified length
 */
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}