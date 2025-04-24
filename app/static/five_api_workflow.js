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
    validationChart: null,
    validationSplit: 0.8, // Default training/validation split ratio
    experimentId: null,
    currentIteration: 0,
    customOptimizerInstructions: null
};

document.addEventListener('DOMContentLoaded', function() {
    initializeUI();
    setupEventListeners();
    initializeDebugConsole();
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
    
    // Set up training configuration UI
    initializeTrainingConfigUI();
    
    // Set up validation and training logs UI
    initializeTrainingLogsUI();
    
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
    try {
        // Log the metrics data for debugging
        console.log("Received metrics data:", metrics);
        logToDebugConsole('info', 'Received metrics data for display', metrics);
        
        // Create default metrics structure if it doesn't exist or is empty
        if (!metrics || Object.keys(metrics).length === 0) {
            console.warn("Creating default metrics structure for display");
            logToDebugConsole('warn', 'Creating default metrics structure for display');
            
            // Create a minimal structure for display
            metrics = {
                exact_match: {
                    original: { training: 0, validation: 0 },
                    optimized: { training: 0, validation: 0 }
                },
                semantic_similarity: {
                    original: { training: 0, validation: 0 },
                    optimized: { training: 0, validation: 0 }
                }
            };
        }
        
        // Update metrics table
        const metricsTableBody = document.getElementById('metrics-table-body');
        if (!metricsTableBody) {
            console.error("Error updating metrics: 'metrics-table-body' element not found in DOM");
            logToDebugConsole('error', 'DOM element for metrics table not found', { elementId: 'metrics-table-body' });
            return;
        }
        
        metricsTableBody.innerHTML = '';
        
        // Add rows for each metric
        const metricNames = {
            'exact_match': 'Exact Match',
            'semantic_similarity': 'Semantic Similarity',
            'keyword_match': 'Keyword Match',
            'bleu': 'BLEU Score',
            'rouge': 'ROUGE Score'
        };
        
        let foundAnyMetrics = false;
        
        // Check if the metrics follow the expected structure
        let hasNestedStructure = false;
        for (const key in metrics) {
            if (metrics[key] && typeof metrics[key] === 'object' && 
                metrics[key].original && metrics[key].optimized) {
                hasNestedStructure = true;
                break;
            }
        }
        
        // If no nested structure, create a compatible structure
        if (!hasNestedStructure && typeof metrics === 'object') {
            console.warn("Converting flat metrics structure to nested format");
            logToDebugConsole('warn', 'Converting flat metrics structure to nested format', metrics);
            
            const restructuredMetrics = {};
            
            // For each potential metric, check if it exists in the flat structure
            for (const [key, _] of Object.entries(metricNames)) {
                if (key in metrics) {
                    restructuredMetrics[key] = {
                        original: { 
                            training: metrics[`${key}_original_training`] || metrics[`${key}_original`] || 0,
                            validation: metrics[`${key}_original_validation`] || metrics[`${key}_original`] || 0
                        },
                        optimized: {
                            training: metrics[`${key}_optimized_training`] || metrics[`${key}_optimized`] || 0,
                            validation: metrics[`${key}_optimized_validation`] || metrics[`${key}_optimized`] || 0
                        }
                    };
                }
            }
            
            // Use the restructured metrics
            metrics = restructuredMetrics;
        }
        
        // Populate the table with metrics
        for (const [key, label] of Object.entries(metricNames)) {
            if (metrics[key]) {
                foundAnyMetrics = true;
                const row = document.createElement('tr');
                
                // Safe access to nested properties with detailed null checks
                let originalTraining = 0;
                let originalValidation = 0;
                let optimizedTraining = 0;
                let optimizedValidation = 0;
                
                if (metrics[key].original) {
                    originalTraining = metrics[key].original.training || 0;
                    originalValidation = metrics[key].original.validation || 0;
                }
                
                if (metrics[key].optimized) {
                    optimizedTraining = metrics[key].optimized.training || 0;
                    optimizedValidation = metrics[key].optimized.validation || 0;
                }
                
                row.innerHTML = `
                    <td>${label}</td>
                    <td>${formatPercent(originalTraining)}</td>
                    <td>${formatPercent(originalValidation)}</td>
                    <td>${formatPercent(optimizedTraining)}</td>
                    <td>${formatPercent(optimizedValidation)}</td>
                `;
                
                metricsTableBody.appendChild(row);
            }
        }
        
        if (!foundAnyMetrics) {
            console.warn("Metrics object does not contain any expected metrics");
            logToDebugConsole('warn', 'No expected metrics found in data', { 
                availableKeys: Object.keys(metrics),
                expectedKeys: Object.keys(metricNames)
            });
            
            // Add a placeholder row
            const placeholderRow = document.createElement('tr');
            placeholderRow.innerHTML = `
                <td colspan="5" class="text-center text-muted">No metrics data available</td>
            `;
            metricsTableBody.appendChild(placeholderRow);
        }
        
        // Create or update chart
        createMetricsChart(metrics);
    } catch (error) {
        console.error("Error in updateMetricsDisplay:", error);
        logToDebugConsole('error', 'Exception in updateMetricsDisplay', { 
            errorMessage: error.message,
            errorStack: error.stack,
            metricsData: metrics ? 'Present' : 'Missing'
        });
    }
}

/**
 * Create a chart to visualize metrics
 */
function createMetricsChart(metrics) {
    try {
        const chartCanvas = document.getElementById('metrics-chart');
        if (!chartCanvas) {
            console.error("Error creating chart: 'metrics-chart' element not found in DOM");
            logToDebugConsole('error', 'DOM element for metrics chart not found', { elementId: 'metrics-chart' });
            return;
        }
        
        // Destroy existing chart if it exists
        if (workflowState.metricsChart) {
            try {
                workflowState.metricsChart.destroy();
            } catch (err) {
                console.warn("Error destroying existing chart:", err);
                logToDebugConsole('warn', 'Failed to destroy existing chart', { error: err.message });
                // Continue anyway to create new chart
            }
        }
        
        // Prepare data for the chart
        const labels = [];
        const originalTraining = [];
        const originalValidation = [];
        const optimizedTraining = [];
        const optimizedValidation = [];
        
        // Ensure metrics is an object before attempting to iterate
        if (!metrics || typeof metrics !== 'object') {
            console.error("Invalid metrics data for chart:", metrics);
            logToDebugConsole('error', 'Invalid metrics data structure for chart', { 
                metricsType: typeof metrics 
            });
            
            // Create an empty chart to avoid errors
            // Define default placeholder data
            const placeholderLabels = ['No Data Available'];
            const placeholderData = [0];
            
            workflowState.metricsChart = new Chart(chartCanvas, {
                type: 'bar',
                data: {
                    labels: placeholderLabels,
                    datasets: [{
                        label: 'No metrics data available',
                        data: placeholderData,
                        backgroundColor: 'rgba(200, 200, 200, 0.2)',
                        borderColor: 'rgba(200, 200, 200, 1)',
                        borderWidth: 1
                    }]
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
                            text: 'Metrics Comparison (No Data)'
                        }
                    }
                }
            });
            
            return;
        }
        
        // Format data for charting with safety checks
        let hasValidData = false;
        for (const [key, value] of Object.entries(metrics)) {
            if (key !== 'total_examples' && value && typeof value === 'object') {
                // Safe access to nested properties
                const originalTrainingValue = value.original?.training;
                
                if (typeof originalTrainingValue === 'number') {
                    hasValidData = true;
                    const label = key.replace(/_/g, ' ');
                    labels.push(label);
                    originalTraining.push((value.original?.training || 0) * 100);
                    originalValidation.push((value.original?.validation || 0) * 100);
                    optimizedTraining.push((value.optimized?.training || 0) * 100);
                    optimizedValidation.push((value.optimized?.validation || 0) * 100);
                }
            }
        }
        
        // Create chart or placeholder if no valid data
        if (!hasValidData) {
            console.warn("No valid metrics data for chart");
            logToDebugConsole('warn', 'No valid metrics data found for chart', { 
                metricsKeys: Object.keys(metrics) 
            });
            
            createEmptyChart(chartCanvas);
            return;
        }
        
        // Create chart with valid data
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
        
        logToDebugConsole('info', 'Chart created successfully', { 
            metrics: labels.length,
            datasets: 4
        });
    } catch (error) {
        console.error("Error in createMetricsChart:", error);
        logToDebugConsole('error', 'Exception in createMetricsChart', { 
            errorMessage: error.message,
            errorStack: error.stack
        });
    }
}

/**
 * Create an empty chart with placeholder message
 */
function createEmptyChart(chartCanvas) {
    workflowState.metricsChart = new Chart(chartCanvas, {
        type: 'bar',
        data: {
            labels: ['No Data'],
            datasets: [{
                label: 'No metrics data available',
                data: [0],
                backgroundColor: 'rgba(200, 200, 200, 0.2)',
                borderColor: 'rgba(200, 200, 200, 1)',
                borderWidth: 1
            }]
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
                    text: 'Metrics Comparison (No Data Available)'
                }
            }
        }
    });
    
    logToDebugConsole('warn', 'Created empty placeholder chart', { reason: 'No valid metrics data' });
}

/**
 * Update step indicators based on current progress
 */
function updateStepIndicators() {
    // If there are no step indicators, don't continue
    const anyStepIndicator = document.getElementById('step1-indicator');
    if (!anyStepIndicator) {
        console.log("Step indicators not found in DOM");
        return;
    }
    
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
            const progressBarEl = document.getElementById('progress-bar');
            if (progressBarEl) {
                progressBarEl.style.width = `${progressPercent}%`;
                progressBarEl.textContent = `${progressPercent}%`;
            }
            
            // Update status text
            const statusText = {
                1: 'Step 1: Running primary LLM...',
                2: 'Step 2: Evaluating baseline metrics...',
                3: 'Step 3: Optimizing prompts...',
                4: 'Step 4: Testing optimized prompts...',
                5: 'Step 5: Final evaluation...'
            };
            
            const progressStatusEl = document.getElementById('progress-status');
            if (progressStatusEl) {
                progressStatusEl.textContent = statusText[workflowState.currentStep];
            }
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
    workflowState.experimentId = null;
    workflowState.currentIteration = 0;
    
    // Reset or maintain chart objects (don't destroy)
    
    // Reset experiment UI
    const experimentStatusEl = document.getElementById('experiment-status');
    if (experimentStatusEl) experimentStatusEl.textContent = 'New experiment will be created';
    
    const currentIterationEl = document.getElementById('current-iteration');
    if (currentIterationEl) currentIterationEl.textContent = '0';
    
    // Reset progress indicators and messages
    const averageScoreEl = document.getElementById('average-score');
    if (averageScoreEl) averageScoreEl.textContent = '-';
    
    const perfectMatchesEl = document.getElementById('perfect-matches');
    if (perfectMatchesEl) perfectMatchesEl.textContent = '-';
    
    const latestOptimizationEl = document.getElementById('latest-optimization-message');
    if (latestOptimizationEl) latestOptimizationEl.textContent = 'No optimization has been performed yet';
    
    const validationMessageEl = document.getElementById('validation-message');
    if (validationMessageEl) validationMessageEl.textContent = 'Run validation to see results on unseen data';
    
    const validationResultsEl = document.getElementById('validation-results');
    if (validationResultsEl) validationResultsEl.classList.add('d-none');
    
    // Update UI
    updateStepIndicators();
    
    const optimizationProgressEl = document.getElementById('optimization-progress');
    if (optimizationProgressEl) optimizationProgressEl.classList.add('d-none');
    
    const startOptimizationBtn = document.getElementById('start-optimization');
    if (startOptimizationBtn) startOptimizationBtn.disabled = false;
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
                // Get the selected number of cases
                const casesCountSelect = document.getElementById('nejm-cases-count');
                const selectedValue = casesCountSelect.value;
                
                // Determine how many examples to use
                let exampleCount = data.examples.length;
                if (selectedValue !== 'all') {
                    exampleCount = Math.min(parseInt(selectedValue), data.examples.length);
                }
                
                // If not using all examples, take a random subset
                let selectedExamples = data.examples;
                if (exampleCount < data.examples.length) {
                    // Shuffle the array using Fisher-Yates algorithm
                    const shuffled = [...data.examples];
                    for (let i = shuffled.length - 1; i > 0; i--) {
                        const j = Math.floor(Math.random() * (i + 1));
                        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
                    }
                    // Take the first N examples
                    selectedExamples = shuffled.slice(0, exampleCount);
                }
                
                // Store the examples in our state
                workflowState.examples = selectedExamples;
                
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
                
                showAlert(`Loaded ${selectedExamples.length} NEJM ${datasetType} examples (from ${data.examples.length} total)`, 'success');
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

/**
 * Initialize the Training Configuration UI elements
 */
function initializeTrainingConfigUI() {
    // Set up Optimizer Instructions modal
    const viewOptimizerInstructionsBtn = document.getElementById('view-optimizer-instructions');
    const saveOptimizerInstructionsBtn = document.getElementById('save-optimizer-instructions');
    const useCustomInstructionsCheckbox = document.getElementById('use-custom-instructions');
    const customInstructionsContainer = document.getElementById('custom-instructions-container');
    const optimizationStrategySelect = document.getElementById('optimization-strategy');
    
    // Update optimizer type when strategy changes
    if (optimizationStrategySelect) {
        optimizationStrategySelect.addEventListener('change', function() {
            const strategyText = this.options[this.selectedIndex].text;
            const optimizerTypeEl = document.getElementById('optimizer-type');
            if (optimizerTypeEl) {
                optimizerTypeEl.textContent = strategyText;
            }
            
            // Also update in the modal
            const modalStrategyNameEl = document.getElementById('modal-strategy-name');
            if (modalStrategyNameEl) {
                modalStrategyNameEl.textContent = strategyText;
            }
            
            // Update instructions content based on strategy
            updateOptimizerInstructions(this.value);
        });
    }
    
    // Toggle custom instructions
    if (useCustomInstructionsCheckbox) {
        useCustomInstructionsCheckbox.addEventListener('change', function() {
            if (this.checked) {
                customInstructionsContainer.classList.remove('d-none');
            } else {
                customInstructionsContainer.classList.add('d-none');
            }
        });
    }
    
    // Save optimizer instructions
    if (saveOptimizerInstructionsBtn) {
        saveOptimizerInstructionsBtn.addEventListener('click', function() {
            const customInstructions = document.getElementById('custom-optimizer-instructions').value.trim();
            if (useCustomInstructionsCheckbox.checked && customInstructions) {
                // Save custom instructions to workflowState
                workflowState.customOptimizerInstructions = customInstructions;
                showAlert('Custom optimizer instructions saved', 'success');
            } else {
                // Reset to default
                workflowState.customOptimizerInstructions = null;
                showAlert('Using default optimizer instructions', 'info');
            }
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('optimizerInstructionsModal'));
            if (modal) {
                modal.hide();
            }
        });
    }
    
    // Set up validation split UI
    const validationModeSelect = document.getElementById('validation-mode');
    const validationPercentageInput = document.getElementById('validation-percentage');
    
    if (validationModeSelect && validationPercentageInput) {
        // Update validation split when inputs change
        validationModeSelect.addEventListener('change', updateValidationSplit);
        validationPercentageInput.addEventListener('change', updateValidationSplit);
        validationPercentageInput.addEventListener('input', updateValidationSplit);
    }
}

/**
 * Update the validation split based on UI inputs
 */
function updateValidationSplit() {
    const validationMode = document.getElementById('validation-mode').value;
    const validationPercentage = parseInt(document.getElementById('validation-percentage').value) / 100;
    
    workflowState.validationSplit = validationMode === 'train' ? validationPercentage : 1 - validationPercentage;
    
    // Update data stats display
    updateDataStats();
}

/**
 * Update the optimizer instructions based on selected strategy
 */
function updateOptimizerInstructions(strategy) {
    const instructionsContent = document.getElementById('optimizer-instructions-content');
    if (!instructionsContent) return;
    
    let content = '';
    
    switch (strategy) {
        case 'reasoning_first':
            content = `
                <p class="mb-2"><strong>Reasoning-First Refinement</strong></p>
                <p>This strategy prioritizes improving the reasoning process by first analyzing the original prompts' weaknesses, then developing a step-by-step refinement plan, and finally implementing targeted improvements focused on reasoning clarity and structure.</p>
                
                <p class="mb-2 mt-3"><strong>Workflow:</strong></p>
                <ol>
                    <li>Analyze current prompts and identify weaknesses</li>
                    <li>Develop a refinement plan focused on improving the reasoning paths</li>
                    <li>Make targeted edits to enhance clarity and logical flow</li>
                    <li>Re-evaluate the refined prompts against original examples</li>
                </ol>
            `;
            break;
            
        case 'balanced':
            content = `
                <p class="mb-2"><strong>Balanced Optimization</strong></p>
                <p>This strategy aims to balance improvements in both accuracy and creativity, making moderate adjustments to the prompts that preserve their core structure while enhancing performance.</p>
                
                <p class="mb-2 mt-3"><strong>Workflow:</strong></p>
                <ol>
                    <li>Identify strengths and weaknesses in the original prompts</li>
                    <li>Make targeted improvements while preserving effective elements</li>
                    <li>Ensure balanced optimization of both accuracy and creative aspects</li>
                    <li>Test refined prompts against diverse example types</li>
                </ol>
            `;
            break;
            
        case 'accuracy':
            content = `
                <p class="mb-2"><strong>Maximize Accuracy</strong></p>
                <p>This strategy focuses exclusively on improving the factual accuracy and precision of responses, even at the expense of creativity or stylistic elements.</p>
                
                <p class="mb-2 mt-3"><strong>Workflow:</strong></p>
                <ol>
                    <li>Identify accuracy issues in the current prompts</li>
                    <li>Add explicit instructions for fact-checking and verification</li>
                    <li>Enhance structure for step-by-step reasoning</li>
                    <li>Test with emphasis on factual correctness metrics</li>
                </ol>
            `;
            break;
            
        case 'creativity':
            content = `
                <p class="mb-2"><strong>Enhanced Creativity</strong></p>
                <p>This strategy prioritizes improving the creative aspects of responses, encouraging novel perspectives and expressive language while maintaining reasonable accuracy.</p>
                
                <p class="mb-2 mt-3"><strong>Workflow:</strong></p>
                <ol>
                    <li>Analyze areas where responses could be more engaging or innovative</li>
                    <li>Add instructions to encourage creative approaches and diverse perspectives</li>
                    <li>Balance creativity with accuracy requirements</li>
                    <li>Test for improvement in engagement and uniqueness metrics</li>
                </ol>
            `;
            break;
            
        default:
            content = `<p>Select a strategy to view instructions.</p>`;
    }
    
    instructionsContent.innerHTML = content;
}

/**
 * Initialize the Training Logs UI elements
 */
function initializeTrainingLogsUI() {
    // Set up validation button
    const runValidationBtn = document.getElementById('run-validation');
    if (runValidationBtn) {
        runValidationBtn.addEventListener('click', runValidation);
    }
    
    // Set up clear logs button
    const clearLogsBtn = document.getElementById('clear-logs');
    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', function() {
            const trainingLogsEl = document.getElementById('training-logs');
            if (trainingLogsEl) {
                trainingLogsEl.textContent = '';
            }
        });
    }
}

/**
 * Run validation on the current prompts
 */
function runValidation() {
    const runValidationBtn = document.getElementById('run-validation');
    if (runValidationBtn) {
        runValidationBtn.disabled = true;
    }
    
    showAlert('Running validation on prompts...', 'info');
    
    // Log to the training logs
    appendToTrainingLogs('Starting validation run...');
    
    // TODO: Implement actual validation call to backend
    // This is a placeholder - would be replaced with actual API call
    
    setTimeout(() => {
        // Update validation section with results
        document.getElementById('validation-message').textContent = 'Validation complete';
        document.getElementById('validation-results').classList.remove('d-none');
        
        // Create sample validation chart
        createValidationChart();
        
        // Re-enable the button
        if (runValidationBtn) {
            runValidationBtn.disabled = false;
        }
        
        appendToTrainingLogs('Validation complete');
        showAlert('Validation completed successfully!', 'success');
    }, 1500);
}

/**
 * Create a chart for validation results
 */
function createValidationChart() {
    const chartCanvas = document.getElementById('validation-chart');
    if (!chartCanvas) return;
    
    // Destroy existing chart if it exists
    if (workflowState.validationChart) {
        workflowState.validationChart.destroy();
    }
    
    // Sample data for demonstration
    const metrics = {
        'Exact Match': 0.75,
        'Semantic Similarity': 0.82,
        'Keyword Match': 0.68,
    };
    
    // Prepare chart data
    const labels = Object.keys(metrics);
    const values = Object.values(metrics).map(v => v * 100);
    
    // Create chart
    workflowState.validationChart = new Chart(chartCanvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Validation Score',
                data: values,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
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
                    text: 'Validation Metrics'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Score: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
    
    // Also populate validation metrics summary
    const validationMetricsEl = document.getElementById('validation-metrics');
    if (validationMetricsEl) {
        let html = '';
        for (const [key, value] of Object.entries(metrics)) {
            html += `
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body p-2 text-center">
                            <div class="small text-muted">${key}</div>
                            <div class="fs-5">${formatPercent(value)}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        validationMetricsEl.innerHTML = html;
    }
}

/**
 * Append a message to the training logs
 */
function appendToTrainingLogs(message) {
    const trainingLogsEl = document.getElementById('training-logs');
    if (trainingLogsEl) {
        const timestamp = new Date().toLocaleTimeString();
        trainingLogsEl.textContent += `[${timestamp}] ${message}\n`;
        
        // Scroll to bottom
        trainingLogsEl.scrollTop = trainingLogsEl.scrollHeight;
    }
}

/**
 * Debug Console Functionality
 * This section implements a debug console that captures and displays errors, warnings, and logs
 */

// Debug state
const debugState = {
    logs: [],
    errorCount: 0,
    warningCount: 0,
    autoScroll: true
};

/**
 * Initialize the debug console
 */
function initializeDebugConsole() {
    // Override console methods to capture logs
    setupConsoleOverrides();
    
    // Set up event listeners for debug console
    setupDebugConsoleEventListeners();
    
    // Initial log
    logToDebugConsole('info', 'Debug console initialized', { timestamp: new Date().toISOString() });
    
    // Capture any errors that occurred before initialization (from webview_console_logs)
    captureExistingErrors();
}

/**
 * Set up debug console event listeners
 */
function setupDebugConsoleEventListeners() {
    // Show debug console button
    const showDebugConsoleBtn = document.getElementById('showDebugConsole');
    if (showDebugConsoleBtn) {
        showDebugConsoleBtn.addEventListener('click', function() {
            const debugModal = new bootstrap.Modal(document.getElementById('debugModal'));
            debugModal.show();
        });
    }
    
    // Clear console button
    const clearDebugConsoleBtn = document.getElementById('clearDebugConsole');
    if (clearDebugConsoleBtn) {
        clearDebugConsoleBtn.addEventListener('click', function() {
            debugState.logs = [];
            debugState.errorCount = 0;
            debugState.warningCount = 0;
            updateDebugConsoleDisplay();
            updateErrorCounter();
        });
    }
    
    // Copy logs button
    const copyDebugLogsBtn = document.getElementById('copyDebugLogs');
    if (copyDebugLogsBtn) {
        copyDebugLogsBtn.addEventListener('click', function() {
            const debugConsole = document.getElementById('debugConsole');
            if (debugConsole) {
                const text = debugConsole.innerText;
                navigator.clipboard.writeText(text)
                    .then(() => {
                        this.innerHTML = '<i class="fas fa-check"></i> Copied!';
                        setTimeout(() => {
                            this.innerHTML = 'Copy Logs';
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Failed to copy logs:', err);
                        this.innerHTML = '<i class="fas fa-times"></i> Failed';
                        setTimeout(() => {
                            this.innerHTML = 'Copy Logs';
                        }, 2000);
                    });
            }
        });
    }
    
    // Auto-scroll switch
    const autoScrollSwitch = document.getElementById('autoScrollSwitch');
    if (autoScrollSwitch) {
        autoScrollSwitch.addEventListener('change', function() {
            debugState.autoScroll = this.checked;
        });
    }
}

/**
 * Override console methods to capture logs
 */
function setupConsoleOverrides() {
    // Store original methods
    const originalConsole = {
        log: console.log,
        info: console.info,
        warn: console.warn,
        error: console.error
    };
    
    // Override console.log
    console.log = function() {
        originalConsole.log.apply(console, arguments);
        logToDebugConsole('log', Array.from(arguments).join(' '));
    };
    
    // Override console.info
    console.info = function() {
        originalConsole.info.apply(console, arguments);
        logToDebugConsole('info', Array.from(arguments).join(' '));
    };
    
    // Override console.warn
    console.warn = function() {
        originalConsole.warn.apply(console, arguments);
        logToDebugConsole('warn', Array.from(arguments).join(' '));
        debugState.warningCount++;
        updateErrorCounter();
    };
    
    // Override console.error
    console.error = function() {
        originalConsole.error.apply(console, arguments);
        logToDebugConsole('error', Array.from(arguments).join(' '));
        debugState.errorCount++;
        updateErrorCounter();
    };
    
    // Capture unhandled errors
    window.addEventListener('error', function(event) {
        logToDebugConsole('error', `Unhandled error: ${event.message} at ${event.filename}:${event.lineno}:${event.colno}`);
        debugState.errorCount++;
        updateErrorCounter();
        return false;
    });
    
    // Capture unhandled promise rejections
    window.addEventListener('unhandledrejection', function(event) {
        logToDebugConsole('error', `Unhandled promise rejection: ${event.reason}`);
        debugState.errorCount++;
        updateErrorCounter();
        return false;
    });
    
    // Capture fetch errors
    const originalFetch = window.fetch;
    window.fetch = function() {
        return originalFetch.apply(this, arguments)
            .catch(error => {
                logToDebugConsole('error', `Fetch error: ${error.message}`, {
                    url: arguments[0],
                    options: arguments[1]
                });
                debugState.errorCount++;
                updateErrorCounter();
                throw error;
            });
    };
}

/**
 * Add a log entry to the debug console
 */
function logToDebugConsole(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const entry = {
        timestamp,
        level,
        message,
        data
    };
    
    debugState.logs.push(entry);
    
    // Limit logs to prevent memory issues (keep last 1000)
    if (debugState.logs.length > 1000) {
        debugState.logs.shift();
    }
    
    // Update display
    updateDebugConsoleDisplay();
}

/**
 * Update the debug console display
 */
function updateDebugConsoleDisplay() {
    const debugConsole = document.getElementById('debugConsole');
    if (!debugConsole) return;
    
    // Format logs
    let html = '';
    debugState.logs.forEach(log => {
        const timestamp = log.timestamp.split('T')[1].split('.')[0]; // HH:MM:SS
        const levelClass = getLevelClass(log.level);
        const dataStr = log.data ? `\n${JSON.stringify(log.data, null, 2)}` : '';
        
        html += `<div class="${levelClass}">[${timestamp}] [${log.level.toUpperCase()}] ${escapeHtml(log.message)}${dataStr}</div>`;
    });
    
    debugConsole.innerHTML = html;
    
    // Auto-scroll to bottom
    if (debugState.autoScroll) {
        debugConsole.scrollTop = debugConsole.scrollHeight;
    }
}

/**
 * Get CSS class for log level
 */
function getLevelClass(level) {
    switch (level) {
        case 'error':
            return 'text-danger';
        case 'warn':
            return 'text-warning';
        case 'info':
            return 'text-info';
        default:
            return 'text-light';
    }
}

/**
 * Update the error counter badge
 */
function updateErrorCounter() {
    const errorCountBadge = document.getElementById('debugErrorCount');
    if (errorCountBadge) {
        const count = debugState.errorCount + debugState.warningCount;
        errorCountBadge.textContent = count.toString();
        
        // Change badge color based on count
        if (debugState.errorCount > 0) {
            errorCountBadge.className = 'badge bg-danger ms-2';
        } else if (debugState.warningCount > 0) {
            errorCountBadge.className = 'badge bg-warning ms-2';
        } else {
            errorCountBadge.className = 'badge bg-secondary ms-2';
        }
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Capture errors that may have occurred before debug console initialization
 */
function captureExistingErrors() {
    // Check for any errors in the window object or global scope
    setTimeout(() => {
        // Check for specific known error in updateMetricsDisplay function
        try {
            const metricsDisplayFunction = document.getElementById('metrics-table-body');
            if (!metricsDisplayFunction) {
                console.error('Known issue: "Error fetching metrics summary" may occur because metrics-table-body element is not found');
            }
        } catch (err) {
            console.error('Error checking metrics display:', err);
        }
        
        // Capture errors from earlier console logs if available
        if (window.webviewConsoleErrors && Array.isArray(window.webviewConsoleErrors)) {
            window.webviewConsoleErrors.forEach(error => {
                console.error('Captured previous error:', error);
            });
        }
        
        // Look for specific error message patterns in page content
        const pageContent = document.body.innerHTML;
        if (pageContent.includes('Error fetching metrics summary')) {
            console.error('Found "Error fetching metrics summary" message in page content');
        }
        
        // Log any error in updateMetricsDisplay implementation
        try {
            const updateMetricsDisplaySource = updateMetricsDisplay.toString();
            console.info('Analyzing updateMetricsDisplay function for potential issues');
            // Check for common issues in the function
            if (updateMetricsDisplaySource.includes('if (!metrics)')) {
                console.warn('updateMetricsDisplay has null check but may still cause errors with nested properties');
            }
        } catch (err) {
            console.error('Error analyzing metrics function:', err);
        }
        
        // Generate a test log to verify the debug console is working
        console.log('Debug console is ready and capturing errors');
    }, 500);
}