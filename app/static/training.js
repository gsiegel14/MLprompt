document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const systemPromptEl = document.getElementById('system-prompt');
    const outputPromptEl = document.getElementById('output-prompt');
    const examplesTextEl = document.getElementById('examples-text');
    const csvFileEl = document.getElementById('csv-file');
    const experimentIdEl = document.getElementById('experiment-id');
    const iterationEl = document.getElementById('iteration');
    const maxIterationsEl = document.getElementById('max-iterations');
    const earlyStoppingEl = document.getElementById('early-stopping');
    const patienceEl = document.getElementById('patience');
    const useValidationEl = document.getElementById('use-validation');
    const trainPercentEl = document.getElementById('train-percent');
    const trainPercentDisplayEl = document.getElementById('train-percent-display');
    const startTrainingBtn = document.getElementById('start-training-btn');
    const stopTrainingBtn = document.getElementById('stop-training-btn');
    const savePromptsBtn = document.getElementById('save-prompts-btn');
    const showSamplePromptsBtn = document.getElementById('show-sample-prompts');
    const optimizerPresetEl = document.getElementById('optimizer-preset');
    const optimizerPromptPreviewEl = document.getElementById('optimizer-prompt-preview');
    const editOptimizerBtn = document.getElementById('edit-optimizer-btn');
    const showOptimizerBtn = document.getElementById('show-optimizer-btn');
    const optimizerPromptFullEl = document.getElementById('optimizer-prompt-full');
    const saveOptimizerBtn = document.getElementById('save-optimizer-btn');
    const loadExperimentBtn = document.getElementById('load-experiment-btn');
    const viewDetailsBtn = document.getElementById('view-details-btn');
    const clearLogsBtn = document.getElementById('clear-logs-btn');
    const validateBtn = document.getElementById('validate-btn');
    const loadNejmTrainBtn = document.getElementById('load-nejm-train-btn');
    const loadNejmValidationBtn = document.getElementById('load-nejm-validation-btn');
    const trainingLogsEl = document.getElementById('training-logs');
    const trainingProgressEl = document.getElementById('training-progress');
    const currentScoreEl = document.getElementById('current-score');
    const perfectMatchesEl = document.getElementById('perfect-matches');
    const optimizationStatusEl = document.getElementById('optimization-status');
    const optimizationResultsEl = document.getElementById('optimization-results');
    const beforeScoreEl = document.getElementById('before-score');
    const afterScoreEl = document.getElementById('after-score');
    const optimizerReasoningEl = document.getElementById('optimizer-reasoning');
    const dataStatsEl = document.getElementById('data-stats');
    const spinner = document.getElementById('spinner');
    
    // Chart.js setup
    let metricsChart;
    setupChart();
    
    // State variables
    let trainingInProgress = false;
    let currentExperimentId = null;
    let currentIteration = 0;
    let optimizerPrompt = '';
    let trainingData = [];
    let validationSplit = 0.8;
    let trainingHistory = [];
    
    // Initialize UI
    initializeUI();
    
    // Event listeners
    startTrainingBtn.addEventListener('click', startTraining);
    stopTrainingBtn.addEventListener('click', stopTraining);
    savePromptsBtn.addEventListener('click', savePrompts);
    showSamplePromptsBtn.addEventListener('click', showSamplePrompts);
    editOptimizerBtn.addEventListener('click', showOptimizerModal);
    showOptimizerBtn.addEventListener('click', showOptimizerModal);
    saveOptimizerBtn.addEventListener('click', saveOptimizerPrompt);
    loadExperimentBtn.addEventListener('click', showExperimentModal);
    viewDetailsBtn.addEventListener('click', viewFullDetails);
    clearLogsBtn.addEventListener('click', clearLogs);
    validateBtn.addEventListener('click', validatePrompts);
    csvFileEl.addEventListener('change', handleCSVUpload);
    trainPercentEl.addEventListener('input', updateTrainPercentDisplay);
    optimizerPresetEl.addEventListener('change', updateOptimizerPreset);
    loadNejmTrainBtn.addEventListener('click', () => loadNejmDataset('train'));
    loadNejmValidationBtn.addEventListener('click', () => loadNejmDataset('validation'));
    
    // Initialize UI
    function initializeUI() {
        // Load optimizer prompt
        fetchOptimizerPrompt();
        
        // Set up train percent display
        updateTrainPercentDisplay();
        
        // Load advanced medical prompts by default
        loadMedicalPrompts();
        
        // Sample examples data
        examplesTextEl.value = 'What is the capital of France?,Paris\nHow many planets are in our solar system?,8\nWhat is the boiling point of water in Celsius?,100';
        
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        
        updateDataStats();
    }
    
    // Chart setup
    function setupChart() {
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        metricsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Average Score',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Perfect Match %',
                        data: [],
                        borderColor: 'rgba(153, 102, 255, 1)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    // Update chart with new data
    function updateChart(history) {
        const labels = history.map((item, index) => `Iteration ${index}`);
        const avgScores = history.map(item => item.metrics.avg_score * 100);
        const perfectMatches = history.map(item => item.metrics.perfect_match_percent);
        
        metricsChart.data.labels = labels;
        metricsChart.data.datasets[0].data = avgScores;
        metricsChart.data.datasets[1].data = perfectMatches;
        metricsChart.update();
    }
    
    // Update train percent display
    function updateTrainPercentDisplay() {
        const value = trainPercentEl.value;
        trainPercentDisplayEl.textContent = value;
        validationSplit = value / 100;
        updateDataStats();
    }
    
    // Update data statistics
    function updateDataStats() {
        const examples = parseExamples();
        trainingData = examples;
        
        const totalCount = examples.length;
        const trainCount = Math.floor(totalCount * validationSplit);
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
        
        dataStatsEl.innerHTML = statsHTML;
    }
    
    // Parse examples from textarea
    function parseExamples() {
        const text = examplesTextEl.value.trim();
        if (!text) return [];
        
        const lines = text.split('\n');
        const examples = [];
        
        for (const line of lines) {
            const [userInput, groundTruth] = line.split(',').map(s => s.trim());
            if (userInput && groundTruth) {
                examples.push({
                    user_input: userInput,
                    ground_truth_output: groundTruth
                });
            }
        }
        
        return examples;
    }
    
    // Handle CSV file upload
    function handleCSVUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        showSpinner();
        fetch('/upload_csv', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else if (data.examples && data.examples.length > 0) {
                // Format examples as CSV lines
                const formattedExamples = data.examples.map(ex => 
                    `${ex.user_input},${ex.ground_truth_output}`
                ).join('\n');
                
                examplesTextEl.value = formattedExamples;
                updateDataStats();
                showAlert(`Loaded ${data.examples.length} examples from CSV`, 'success');
            } else {
                showAlert('No valid examples found in CSV', 'warning');
            }
        })
        .catch(error => {
            console.error('Error uploading CSV:', error);
            showAlert('Error uploading CSV file', 'danger');
        })
        .finally(() => {
            hideSpinner();
            csvFileEl.value = ''; // Reset the file input
        });
    }
    
    // Fetch optimizer prompt
    function fetchOptimizerPrompt() {
        fetch('/optimizer_prompt')
            .then(response => response.json())
            .then(data => {
                if (data.optimizer_prompt) {
                    optimizerPrompt = data.optimizer_prompt;
                    // Show preview (first 100 chars)
                    optimizerPromptPreviewEl.value = optimizerPrompt.substring(0, 100) + '...';
                    optimizerPromptFullEl.value = optimizerPrompt;
                }
            })
            .catch(error => {
                console.error('Error fetching optimizer prompt:', error);
            });
    }
    
    // Show optimizer modal
    function showOptimizerModal() {
        optimizerPromptFullEl.value = optimizerPrompt;
        const modal = new bootstrap.Modal(document.getElementById('optimizer-modal'));
        modal.show();
    }
    
    // Save optimizer prompt
    function saveOptimizerPrompt() {
        const newPrompt = optimizerPromptFullEl.value.trim();
        if (!newPrompt) {
            showAlert('Optimizer prompt cannot be empty', 'warning');
            return;
        }
        
        showSpinner();
        fetch('/optimizer_prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                optimizer_prompt: newPrompt
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                optimizerPrompt = newPrompt;
                optimizerPromptPreviewEl.value = newPrompt.substring(0, 100) + '...';
                
                // Close the modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('optimizer-modal'));
                modal.hide();
                
                showAlert('Optimizer prompt saved successfully', 'success');
            }
        })
        .catch(error => {
            console.error('Error saving optimizer prompt:', error);
            showAlert('Error saving optimizer prompt', 'danger');
        })
        .finally(() => {
            hideSpinner();
        });
    }
    
    // Update optimizer preset
    function updateOptimizerPreset() {
        const preset = optimizerPresetEl.value;
        // In a real implementation, we would load different presets
        // For now, just show a message
        log(`Changed optimizer strategy to ${preset}`);
    }
    
    // Save prompts
    function savePrompts() {
        const systemPrompt = systemPromptEl.value.trim();
        const outputPrompt = outputPromptEl.value.trim();
        
        if (!systemPrompt || !outputPrompt) {
            showAlert('Both system prompt and output prompt are required', 'warning');
            return;
        }
        
        showSpinner();
        fetch('/save_prompts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                system_prompt: systemPrompt,
                output_prompt: outputPrompt
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                showAlert('Prompts saved successfully', 'success');
            }
        })
        .catch(error => {
            console.error('Error saving prompts:', error);
            showAlert('Error saving prompts', 'danger');
        })
        .finally(() => {
            hideSpinner();
        });
    }
    
    // Load advanced medical prompts
    function loadMedicalPrompts() {
        showSpinner();
        
        // Use the nejm_prompts endpoint which already has the logic to load these files
        fetch('/load_dataset?type=nejm_prompts')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                } else if (data.prompts) {
                    systemPromptEl.value = data.prompts.system_prompt || '';
                    outputPromptEl.value = data.prompts.output_prompt || '';
                    showAlert('Medical prompts loaded successfully', 'success');
                } else {
                    showAlert('Failed to load medical prompts', 'warning');
                }
            })
            .catch(error => {
                console.error('Error loading medical prompts:', error);
                showAlert('Error loading medical prompts', 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    // Show sample prompts
    function showSamplePrompts() {
        systemPromptEl.value = 'You are an AI assistant tasked with answering factual questions accurately and concisely. Provide direct, correct answers without unnecessary elaboration.';
        outputPromptEl.value = 'Your response should be factual, concise, and directly answer the question. Avoid explaining your reasoning unless specifically asked.';
        
        showAlert('Sample prompts loaded', 'info');
    }
    
    // Load NEJM dataset (train or validation)
    function loadNejmDataset(datasetType) {
        showSpinner();
        log(`Loading NEJM ${datasetType} dataset...`);
        
        fetch(`/load_dataset?type=nejm_${datasetType}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                } else if (data.examples && data.examples.length > 0) {
                    // Format examples as CSV lines
                    const formattedExamples = data.examples.map(ex => 
                        `${ex.user_input},${ex.ground_truth_output}`
                    ).join('\n');
                    
                    examplesTextEl.value = formattedExamples;
                    updateDataStats();
                    
                    // Update train/validation percentage based on dataset type
                    if (datasetType === 'train') {
                        trainPercentEl.value = 100;
                        trainPercentDisplayEl.textContent = 100;
                        validationSplit = 1.0;
                    } else if (datasetType === 'validation') {
                        trainPercentEl.value = 0;
                        trainPercentDisplayEl.textContent = 0;
                        validationSplit = 0.0;
                    }
                    
                    // Also load corresponding prompts
                    fetch('/load_dataset?type=nejm_prompts')
                        .then(response => response.json())
                        .then(promptData => {
                            if (!promptData.error && promptData.prompts) {
                                systemPromptEl.value = promptData.prompts.system_prompt || systemPromptEl.value;
                                outputPromptEl.value = promptData.prompts.output_prompt || outputPromptEl.value;
                                log('Loaded NEJM specialized prompts');
                            }
                        })
                        .catch(error => console.error('Error loading NEJM prompts:', error));
                    
                    updateDataStats();
                    showAlert(`Loaded ${data.examples.length} NEJM ${datasetType} examples`, 'success');
                } else {
                    showAlert(`No NEJM ${datasetType} examples found`, 'warning');
                }
            })
            .catch(error => {
                console.error(`Error loading NEJM ${datasetType} dataset:`, error);
                showAlert(`Error loading NEJM ${datasetType} dataset`, 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    // Show experiment modal
    function showExperimentModal() {
        // Fetch experiments
        showSpinner();
        fetch('/experiments')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                } else if (data.experiments && data.experiments.length > 0) {
                    populateExperimentsTable(data.experiments);
                    const modal = new bootstrap.Modal(document.getElementById('experiment-modal'));
                    modal.show();
                } else {
                    showAlert('No experiments found', 'info');
                }
            })
            .catch(error => {
                console.error('Error fetching experiments:', error);
                showAlert('Error loading experiments', 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    // Populate experiments table
    function populateExperimentsTable(experiments) {
        const tableBody = document.querySelector('#experiments-table tbody');
        tableBody.innerHTML = '';
        
        experiments.forEach(exp => {
            const date = new Date(exp.timestamp * 1000);
            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${exp.experiment_id}</td>
                <td>${exp.iteration + 1}</td>
                <td>${(exp.metrics.avg_score * 100).toFixed(1)}%</td>
                <td>${formattedDate}</td>
                <td>
                    <button class="btn btn-sm btn-primary load-experiment" data-id="${exp.experiment_id}">Load</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
        
        // Add event listeners to load buttons
        document.querySelectorAll('.load-experiment').forEach(btn => {
            btn.addEventListener('click', function() {
                const expId = this.getAttribute('data-id');
                loadExperiment(expId);
                
                // Close the modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('experiment-modal'));
                modal.hide();
            });
        });
    }
    
    // Load experiment
    function loadExperiment(experimentId) {
        showSpinner();
        fetch(`/experiments/${experimentId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                } else if (data.iterations && data.iterations.length > 0) {
                    // Set current experiment
                    currentExperimentId = experimentId;
                    experimentIdEl.value = experimentId;
                    
                    // Get latest iteration
                    const latestIteration = data.iterations[data.iterations.length - 1];
                    currentIteration = latestIteration.iteration;
                    iterationEl.value = currentIteration;
                    
                    // Load prompts
                    systemPromptEl.value = latestIteration.system_prompt;
                    outputPromptEl.value = latestIteration.output_prompt;
                    
                    // Load metrics
                    updateMetricsDisplay(latestIteration.metrics);
                    
                    // Build history for chart
                    trainingHistory = data.iterations;
                    updateChart(trainingHistory);
                    
                    // Show the latest optimization if available
                    if (data.iterations.length > 1) {
                        const previousIteration = data.iterations[data.iterations.length - 2];
                        showOptimizationComparison(previousIteration, latestIteration);
                    }
                    
                    showAlert(`Loaded experiment ${experimentId} (${data.iterations.length} iterations)`, 'success');
                    log(`Loaded experiment ${experimentId} with ${data.iterations.length} iterations`);
                } else {
                    showAlert(`No iterations found for experiment ${experimentId}`, 'warning');
                }
            })
            .catch(error => {
                console.error('Error loading experiment:', error);
                showAlert('Error loading experiment', 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    // View full details
    function viewFullDetails() {
        // In a real implementation, this would navigate to a detailed view
        // For now, just show a message
        showAlert('Full experiment details view not implemented in this demo', 'info');
    }
    
    // Clear logs
    function clearLogs() {
        trainingLogsEl.textContent = '';
    }
    
    // Start training
    function startTraining() {
        const systemPrompt = systemPromptEl.value.trim();
        const outputPrompt = outputPromptEl.value.trim();
        const examples = parseExamples();
        const optimizerType = document.getElementById('optimizer-type').value;
        const optimizerStrategy = document.getElementById('optimizer-strategy').value;
        
        if (!systemPrompt || !outputPrompt) {
            showAlert('Both system prompt and output prompt are required', 'warning');
            return;
        }
        
        if (examples.length === 0) {
            showAlert('No valid examples found', 'warning');
            return;
        }
        
        // Update UI
        trainingInProgress = true;
        startTrainingBtn.disabled = true;
        stopTrainingBtn.disabled = false;
        
        log('Starting Two-Stage Training Cycle...');
        log(`System prompt: ${systemPrompt.substring(0, 50)}...`);
        log(`Output prompt: ${outputPrompt.substring(0, 50)}...`);
        log(`Training with ${examples.length} examples`);
        log(`Optimizer type: ${optimizerType}, Strategy: ${optimizerStrategy}`);
        
        // Reset progress bar
        updateTrainingProgress(0, maxIterationsEl.value);
        
        // Clear chart data
        metricsChart.data.labels = [];
        metricsChart.data.datasets[0].data = [];
        metricsChart.data.datasets[1].data = [];
        metricsChart.update();
        
        // Start the training process with the two-stage workflow
        showSpinner();
        fetch('/two_stage_train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                system_prompt: systemPrompt,
                output_prompt: outputPrompt,
                examples_content: examplesTextEl.value,
                max_iterations: parseInt(maxIterationsEl.value),
                optimizer_strategy: optimizerStrategy,
                optimizer_type: optimizerType
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
                log(`Error: ${data.error}`);
            } else {
                // Update experiment ID and iteration
                currentExperimentId = data.experiment_id;
                experimentIdEl.value = data.experiment_id;
                currentIteration = data.iterations || 0;
                iterationEl.value = currentIteration;
                
                // Log results
                log(`Completed ${data.iterations} iterations for experiment ${data.experiment_id}`);
                log(`Best score achieved: ${(data.best_score * 100).toFixed(1)}% at iteration ${data.best_iteration}`);
                
                // If optimization was performed
                if (data.optimized && data.optimized.metrics) {
                    log(`Optimized metrics: Avg score ${(data.optimized.metrics.avg_score * 100).toFixed(1)}%, Perfect matches: ${data.optimized.metrics.perfect_matches}/${data.optimized.metrics.total_examples}`);
                    log(`Improvement: ${(data.improvement * 100).toFixed(1)}%`);
                    
                    // Update prompts with optimized versions
                    systemPromptEl.value = data.optimized.system_prompt;
                    outputPromptEl.value = data.optimized.output_prompt;
                    
                    // Update metrics display
                    updateMetricsDisplay(data.optimized.metrics);
                    
                    // Show optimization comparison
                    showOptimizationComparison(data.initial, data.optimized);
                    
                    // Add to training history for chart
                    trainingHistory.push({
                        iteration: data.optimized_iteration,
                        metrics: data.optimized.metrics,
                        system_prompt: data.optimized.system_prompt,
                        output_prompt: data.optimized.output_prompt
                    });
                } else {
                    // No optimization (perfect score or other reason)
                    log(data.message || 'No optimization performed');
                    
                    // Update metrics display
                    updateMetricsDisplay(data.initial.metrics);
                    
                    // Add to training history for chart
                    trainingHistory.push({
                        iteration: data.initial_iteration,
                        metrics: data.initial.metrics,
                        system_prompt: data.initial.system_prompt,
                        output_prompt: data.initial.output_prompt
                    });
                }
                
                // Update chart
                updateChart(trainingHistory);
                
                // Update progress
                updateTrainingProgress(1, maxIterationsEl.value);
                
                showAlert('Training iteration completed', 'success');
            }
        })
        .catch(error => {
            console.error('Error in training:', error);
            showAlert('Error during training', 'danger');
            log(`Error: ${error.message}`);
        })
        .finally(() => {
            hideSpinner();
            // Reset UI
            trainingInProgress = false;
            startTrainingBtn.disabled = false;
            stopTrainingBtn.disabled = true;
        });
    }
    
    // Update training progress
    function updateTrainingProgress(current, total) {
        const percent = (current / total) * 100;
        trainingProgressEl.style.width = `${percent}%`;
        trainingProgressEl.textContent = `${Math.round(percent)}%`;
        trainingProgressEl.setAttribute('aria-valuenow', percent);
    }
    
    // Stop training
    function stopTraining() {
        if (!trainingInProgress) return;
        
        trainingInProgress = false;
        startTrainingBtn.disabled = false;
        stopTrainingBtn.disabled = true;
        
        log('Training process stopped by user');
        showAlert('Training stopped', 'info');
    }
    
    // Update metrics display
    function updateMetricsDisplay(metrics) {
        currentScoreEl.textContent = (metrics.avg_score * 100).toFixed(1) + '%';
        perfectMatchesEl.textContent = `${metrics.perfect_matches}/${metrics.total_examples}`;
    }
    
    // Show optimization comparison
    function showOptimizationComparison(before, after) {
        optimizationStatusEl.style.display = 'none';
        optimizationResultsEl.style.display = 'block';
        
        beforeScoreEl.textContent = (before.metrics.avg_score * 100).toFixed(1) + '%';
        afterScoreEl.textContent = (after.metrics.avg_score * 100).toFixed(1) + '%';
        
        // Show reasoning if available
        if (after.reasoning) {
            optimizerReasoningEl.innerHTML = `<p>${after.reasoning.replace(/\n/g, '<br>')}</p>`;
        } else {
            optimizerReasoningEl.innerHTML = '<p class="text-muted">No reasoning available</p>';
        }
        
        // Add classes for improvement visualization
        if (after.metrics.avg_score > before.metrics.avg_score) {
            afterScoreEl.classList.add('text-success');
        } else if (after.metrics.avg_score < before.metrics.avg_score) {
            afterScoreEl.classList.add('text-danger');
        }
    }
    
    // Show spinner
    function showSpinner() {
        spinner.style.display = 'flex';
    }
    
    // Hide spinner
    function hideSpinner() {
        spinner.style.display = 'none';
    }
    
    // Show alert
    function showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alert-container');
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        alertContainer.appendChild(alert);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alertInstance = bootstrap.Alert.getOrCreateInstance(alert);
            alertInstance.close();
        }, 5000);
    }
    
    // Log message
    function log(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logMessage = `[${timestamp}] ${message}\n`;
        trainingLogsEl.textContent += logMessage;
        trainingLogsEl.scrollTop = trainingLogsEl.scrollHeight;
    }
    
    // Validate prompts function - tests prompt versions on validation set
    function validatePrompts() {
        if (trainingHistory.length < 2) {
            showAlert('Need at least 2 iterations to validate', 'warning');
            return;
        }
        
        // Get versions to compare
        const promptVersions = [];
        for (let i = 0; i < trainingHistory.length; i++) {
            promptVersions.push(trainingHistory[i].iteration);
        }
        
        log('Starting validation on unseen data...');
        
        // Show spinner
        showSpinner();
        
        // Send validation request
        fetch('/validate_prompts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt_versions: promptVersions
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
                log(`Validation error: ${data.error}`);
            } else {
                log(`Validation complete on ${data.example_count} unseen examples`);
                
                // Display validation results
                let resultsHtml = '<div class="validation-results p-3 border rounded mb-3">';
                resultsHtml += '<h5>Validation Results (Unseen Data)</h5>';
                resultsHtml += '<table class="table table-sm table-striped">';
                resultsHtml += '<thead><tr><th>Version</th><th>Avg Score</th><th>Perfect Matches</th></tr></thead>';
                resultsHtml += '<tbody>';
                
                let bestVersion = null;
                let bestScore = 0;
                
                for (const [version, results] of Object.entries(data.validation_results)) {
                    const avgScore = results.metrics.avg_score * 100;
                    const perfectMatches = results.metrics.perfect_matches;
                    const totalExamples = results.example_count;
                    
                    resultsHtml += `<tr>
                        <td>${version}</td>
                        <td>${avgScore.toFixed(1)}%</td>
                        <td>${perfectMatches}/${totalExamples}</td>
                    </tr>`;
                    
                    if (avgScore > bestScore) {
                        bestScore = avgScore;
                        bestVersion = version;
                    }
                }
                
                resultsHtml += '</tbody></table>';
                
                if (bestVersion) {
                    resultsHtml += `<div class="alert alert-success">
                        <strong>Best Version:</strong> ${bestVersion} (${bestScore.toFixed(1)}%)
                    </div>`;
                }
                
                resultsHtml += '</div>';
                
                // Add to the logs area
                document.querySelector('.validation-container').innerHTML = resultsHtml;
                
                showAlert('Validation complete', 'success');
            }
        })
        .catch(error => {
            console.error('Error in validation:', error);
            showAlert('Error during validation', 'danger');
            log(`Validation error: ${error.message}`);
        })
        .finally(() => {
            hideSpinner();
        });
    }
});