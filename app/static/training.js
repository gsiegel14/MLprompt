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
    const optimizerPresetEl = document.getElementById('optimizer-strategy');
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
    let trainingData = [];  // Store loaded example data
    let optimizerPrompt = '';
    let validationSplit = 0.8;
    let trainingHistory = [];
    let currentComparisonData = null;
    let spinnerTimeout = null; // For handling stuck spinners
    
    // Initialize UI
    initializeUI();
    
    // Event listeners
    startTrainingBtn.addEventListener('click', startTraining);
    stopTrainingBtn.addEventListener('click', stopTraining);
    savePromptsBtn.addEventListener('click', savePrompts);
    showSamplePromptsBtn.addEventListener('click', showSamplePrompts);
    document.getElementById('load-medical-prompts').addEventListener('click', loadMedicalPrompts);
    editOptimizerBtn.addEventListener('click', showOptimizerModal);
    showOptimizerBtn.addEventListener('click', showOptimizerModal);
    saveOptimizerBtn.addEventListener('click', saveOptimizerPrompt);
    loadExperimentBtn.addEventListener('click', showExperimentModal);
    clearLogsBtn.addEventListener('click', clearLogs);
    validateBtn.addEventListener('click', validatePrompts);
    csvFileEl.addEventListener('change', handleCSVUpload);
    trainPercentEl.addEventListener('input', updateTrainPercentDisplay);
    optimizerPresetEl.addEventListener('change', updateOptimizerPreset);
    loadNejmTrainBtn.addEventListener('click', () => loadNejmDataset('train'));
    loadNejmValidationBtn.addEventListener('click', () => loadNejmDataset('validation'));
    document.getElementById('reset-nejm-cache-btn').addEventListener('click', resetNejmCache);
    document.getElementById('apply-optimized-prompts-btn').addEventListener('click', applyOptimizedPrompts);
    document.getElementById('view-latest-comparisons-btn').addEventListener('click', showLatestComparisons);
    
    // Add event listener for view details and compare prompts buttons
    document.addEventListener('click', function(e) {
        if (e.target && e.target.id === 'view-details-btn') {
            viewFullDetails();
        }
        if (e.target && e.target.id === 'compare-prompts-btn') {
            showPromptComparison();
        }
    });
    
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
        
        // Set up event listeners for comparison tab navigation
        document.getElementById('comparison-prev-example').addEventListener('click', () => {
            if (currentExampleIndex > 0) {
                currentExampleIndex--;
                updateComparisonTabView();
            }
        });
        
        document.getElementById('comparison-next-example').addEventListener('click', () => {
            if (currentExampleIndex < currentExamples.length - 1) {
                currentExampleIndex++;
                updateComparisonTabView();
            }
        });
        
        // Set up tab switch listeners
        document.getElementById('comparisons-tab').addEventListener('shown.bs.tab', function (e) {
            updateComparisonTabView();
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
        // If we already have trainingData loaded from API, use that
        if (trainingData && trainingData.length > 0) {
            return trainingData;
        }
        
        // Otherwise try to parse from text area
        const text = examplesTextEl.value.trim();
        if (!text) return [];
        
        // Skip header row if present
        const lines = text.split('\n');
        const startIndex = lines[0].toLowerCase().includes('user_input') ? 1 : 0;
        const examples = [];
        
        // Process each line, handling escaped commas
        for (let i = startIndex; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            try {
                // Handle escaped commas by first splitting on unescaped commas
                let parts = [];
                let currentPart = '';
                let escapeActive = false;
                
                for (let j = 0; j < line.length; j++) {
                    const char = line[j];
                    
                    if (char === '\\' && j + 1 < line.length && line[j + 1] === ',') {
                        // This is an escaped comma, add just the comma
                        currentPart += ',';
                        j++; // Skip the next character (the comma)
                        continue;
                    }
                    
                    if (char === ',' && !escapeActive) {
                        // Unescaped comma, end of part
                        parts.push(currentPart);
                        currentPart = '';
                        continue;
                    }
                    
                    // Regular character
                    currentPart += char;
                }
                
                // Add the last part
                parts.push(currentPart);
                
                const userInput = parts[0].trim();
                const groundTruth = parts[1] ? parts[1].trim() : '';
                
                if (userInput && groundTruth) {
                    examples.push({
                        user_input: userInput,
                        ground_truth_output: groundTruth
                    });
                }
            } catch (e) {
                console.error('Error parsing line:', line, e);
                // Skip this line and continue
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
                    // Store the example data directly
                    trainingData = data.examples;
                    
                    // Format examples as text for display
                    examplesTextEl.value = data.csv_content || '';
                    
                    // Update stats using the direct count from the API
                    const totalCount = data.count;
                    const trainCount = (datasetType === 'train') ? totalCount : 0;
                    const valCount = (datasetType === 'validation') ? totalCount : 0;
                    
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
    
    // Reset NEJM data cache
    function resetNejmCache() {
        showSpinner();
        log('Resetting NEJM data cache...');
        
        fetch('/reset_nejm_cache', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
                log(`Error: ${data.error}`);
            } else {
                log(data.message);
                showAlert(data.message, 'success');
                
                // Run the fix_nejm_data.py script to regenerate the datasets
                log('Regenerating NEJM datasets...');
                return fetch('/regenerate_nejm_data', {
                    method: 'POST'
                });
            }
        })
        .then(response => {
            if (response) return response.json();
        })
        .then(data => {
            if (data) {
                if (data.error) {
                    showAlert(data.error, 'danger');
                    log(`Error: ${data.error}`);
                } else {
                    log(data.message);
                    showAlert(data.message + '. Try loading the NEJM datasets again.', 'success');
                }
            }
        })
        .catch(error => {
            console.error('Error resetting NEJM cache:', error);
            showAlert('Error resetting NEJM cache', 'danger');
            log(`Error resetting NEJM cache: ${error.message}`);
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
    
    // Show Model Results Modal
    function showModelResults(data) {
        // Prepare the modal with data
        populateModelResultsModal(data);
        
        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById('model-results-modal'));
        modal.show();
        
        // Remove any existing event listeners to prevent duplicates
        const comparisonsTab = document.getElementById('comparisons-tab');
        const newComparisonsTab = comparisonsTab.cloneNode(true);
        comparisonsTab.parentNode.replaceChild(newComparisonsTab, comparisonsTab);
        
        // Add event listener for the comparisons tab to ensure it's populated when clicked
        newComparisonsTab.addEventListener('shown.bs.tab', function (e) {
            console.log("Comparison tab shown");
            updateComparisonTabView();
        });
        
        // Force update if the comparisons tab is already active
        if (newComparisonsTab.classList.contains('active')) {
            updateComparisonTabView();
        }
    }
    
    // Populate Model Results Modal
    function populateModelResultsModal(data) {
        // Clear previous content
        document.getElementById('examplesAccordion').innerHTML = '';
        document.getElementById('error-analysis-table').innerHTML = '';
        
        // Update global examples for the comparison view
        currentExamples = data.examples || [];
        currentExampleIndex = 0;
        
        // Initialize comparison tab view
        const noComparisonExamples = document.getElementById('no-comparison-examples');
        const comparisonContent = document.getElementById('comparison-content');
        
        if (currentExamples && currentExamples.length > 0) {
            noComparisonExamples.style.display = 'none';
            comparisonContent.style.display = 'block';
            
            // Auto-select the comparisons tab to show it first
            // This makes the response comparisons more prominent
            setTimeout(() => {
                const comparisonsTab = document.getElementById('comparisons-tab');
                // Create a Bootstrap tab instance and show it
                const tab = new bootstrap.Tab(comparisonsTab);
                tab.show();
                // Also update the comparison view
                updateComparisonTabView();
            }, 300); // Small delay to ensure the modal is fully loaded
        } else {
            noComparisonExamples.style.display = 'block';
            comparisonContent.style.display = 'none';
        }
        
        // Show/hide no examples message
        const noExamplesMessage = document.getElementById('no-examples-message');
        
        // Overview tab - performance metrics
        if (data.metrics) {
            document.getElementById('result-avg-score').textContent = (data.metrics.avg_score * 100).toFixed(1) + '%';
            document.getElementById('result-perfect-matches').textContent = `${data.metrics.perfect_matches}/${data.metrics.total_examples}`;
            document.getElementById('result-total-examples').textContent = data.metrics.total_examples;
            
            // Set average latency if available
            const avgLatency = data.metrics.avg_latency || 'N/A';
            document.getElementById('result-avg-latency').textContent = typeof avgLatency === 'number' ? 
                `${avgLatency.toFixed(0)}ms` : avgLatency;
        } else {
            document.getElementById('result-avg-score').textContent = '0%';
            document.getElementById('result-perfect-matches').textContent = '0/0';
            document.getElementById('result-total-examples').textContent = '0';
            document.getElementById('result-avg-latency').textContent = 'N/A';
        }
        
        // Experiment info
        document.getElementById('result-experiment-id').textContent = data.experiment_id || '-';
        document.getElementById('result-iteration').textContent = data.iteration || '-';
        document.getElementById('result-timestamp').textContent = new Date().toLocaleString();
        document.getElementById('result-model').textContent = data.model || 'gemini-1.5-flash';
        document.getElementById('result-batch-size').textContent = data.batch_size || '-';
        document.getElementById('result-optimizer-strategy').textContent = data.optimizer_strategy || '-';
        
        // Examples tab - If we have example data
        if (data.examples && data.examples.length > 0) {
            noExamplesMessage.style.display = 'none';
            
            // Generate example accordions
            const examplesContainer = document.getElementById('examplesAccordion');
            
            data.examples.forEach((example, index) => {
                // Calculate score class
                let scoreClass = 'text-warning';
                if (example.score >= 0.9) {
                    scoreClass = 'text-success';
                } else if (example.score < 0.5) {
                    scoreClass = 'text-danger';
                }
                
                // Create accordion item
                const accordionItem = document.createElement('div');
                accordionItem.className = 'accordion-item';
                accordionItem.dataset.score = example.score;
                accordionItem.dataset.category = example.score >= 0.9 ? 'perfect' : 
                                                (example.score >= 0.5 ? 'partial' : 'failed');
                
                accordionItem.innerHTML = `
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                data-bs-target="#example-${index}" aria-expanded="false" aria-controls="example-${index}">
                            <div class="d-flex w-100 justify-content-between align-items-center">
                                <span>Example #${index + 1}: ${truncateText(example.user_input, 70)}</span>
                                <span class="badge ${scoreClass} ms-2">Score: ${(example.score * 100).toFixed(1)}%</span>
                            </div>
                        </button>
                    </h2>
                    <div id="example-${index}" class="accordion-collapse collapse" data-bs-parent="#examplesAccordion">
                        <div class="accordion-body">
                            <div class="mb-3">
                                <h6 class="fw-bold">Input:</h6>
                                <pre class="p-2 bg-light border rounded">${example.user_input}</pre>
                            </div>
                            <div class="mb-3">
                                <h6 class="fw-bold">Expected Output:</h6>
                                <pre class="p-2 bg-light border rounded">${example.ground_truth_output}</pre>
                            </div>
                            <div class="mb-3">
                                <h6 class="fw-bold">Model Response:</h6>
                                <pre class="p-2 bg-light border rounded ${example.score >= 0.9 ? 'border-success' : (example.score < 0.5 ? 'border-danger' : 'border-warning')}">${example.model_response}</pre>
                            </div>
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <span class="badge ${scoreClass}">Score: ${(example.score * 100).toFixed(1)}%</span>
                                </div>
                                <button class="btn btn-sm btn-primary view-response-details" data-example-index="${index}">
                                    <i class="fa-solid fa-magnifying-glass me-1"></i> Analysis
                                </button>
                            </div>
                        </div>
                    </div>
                `;
                
                examplesContainer.appendChild(accordionItem);
            });
            
            // Add example filter logic
            const exampleFilter = document.getElementById('example-filter');
            exampleFilter.addEventListener('change', function() {
                const filterValue = this.value;
                const examples = document.querySelectorAll('.accordion-item');
                
                examples.forEach(example => {
                    if (filterValue === 'all' || example.dataset.category === filterValue) {
                        example.style.display = 'block';
                    } else {
                        example.style.display = 'none';
                    }
                });
            });
            
            // Set up view toggle buttons
            document.getElementById('view-accordion').addEventListener('click', () => toggleExampleView('accordion'));
            document.getElementById('view-comparison').addEventListener('click', () => toggleExampleView('comparison'));
            
            // Set up navigation buttons for comparison view
            document.getElementById('prev-example').addEventListener('click', () => navigateExample(-1));
            document.getElementById('next-example').addEventListener('click', () => navigateExample(1));
            
            // Initialize metrics charts
            initializeMetricsCharts(data);
            
        } else {
            noExamplesMessage.style.display = 'block';
        }
    }
    
    // Initialize metrics charts
    function initializeMetricsCharts(data) {
        // Score distribution chart
        const scoreCtx = document.getElementById('score-distribution-chart').getContext('2d');
        
        // Create score distribution buckets
        const scoreRanges = [
            { min: 0, max: 0.2, label: '0-20%', color: 'rgba(220, 53, 69, 0.7)' },
            { min: 0.2, max: 0.4, label: '20-40%', color: 'rgba(253, 126, 20, 0.7)' },
            { min: 0.4, max: 0.6, label: '40-60%', color: 'rgba(255, 193, 7, 0.7)' },
            { min: 0.6, max: 0.8, label: '60-80%', color: 'rgba(25, 135, 84, 0.6)' },
            { min: 0.8, max: 1.01, label: '80-100%', color: 'rgba(25, 135, 84, 0.9)' }
        ];
        
        // Calculate scores if we have examples
        let scoreCounts = [0, 0, 0, 0, 0];
        
        if (data.examples && data.examples.length > 0) {
            data.examples.forEach(example => {
                const score = example.score;
                // Find which bucket this score belongs to
                for (let i = 0; i < scoreRanges.length; i++) {
                    if (score >= scoreRanges[i].min && score < scoreRanges[i].max) {
                        scoreCounts[i]++;
                        break;
                    }
                }
            });
            
            // Create and fill error analysis table
            createErrorAnalysisTable(data.examples);
        }
        
        // Create chart
        if (window.scoreDistributionChart) {
            window.scoreDistributionChart.destroy();
        }
        
        window.scoreDistributionChart = new Chart(scoreCtx, {
            type: 'bar',
            data: {
                labels: scoreRanges.map(range => range.label),
                datasets: [{
                    label: 'Number of Examples',
                    data: scoreCounts,
                    backgroundColor: scoreRanges.map(range => range.color),
                    borderColor: scoreRanges.map(range => range.color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw} example(s)`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Create error analysis table
    function createErrorAnalysisTable(examples) {
        // Define error categories and initialize counts
        const errorCategories = {
            'missing_content': { count: 0, examples: [] },
            'irrelevant_content': { count: 0, examples: [] },
            'incorrect_diagnosis': { count: 0, examples: [] },
            'format_error': { count: 0, examples: [] },
            'low_score': { count: 0, examples: [] }
        };
        
        // Analyze each example for error patterns
        examples.forEach((example, index) => {
            // Skip high-scoring examples
            if (example.score >= 0.9) return;
            
            // Check for specific error patterns
            const lowerResponse = example.model_response.toLowerCase();
            const lowerTruth = example.ground_truth_output.toLowerCase();
            
            // Basic analysis based on patterns
            if (example.score < 0.5) {
                errorCategories.low_score.count++;
                errorCategories.low_score.examples.push(index);
            }
            
            if (lowerTruth.length > lowerResponse.length * 1.5) {
                errorCategories.missing_content.count++;
                errorCategories.missing_content.examples.push(index);
            }
            
            // Look for format issues (e.g., missing structured output)
            if ((lowerTruth.includes("diagnosis:") && !lowerResponse.includes("diagnosis:")) ||
                (lowerTruth.includes("differential:") && !lowerResponse.includes("differential:"))) {
                errorCategories.format_error.count++;
                errorCategories.format_error.examples.push(index);
            }
            
            // Diagnosis errors (simplified detection)
            if (example.score < 0.8 && example.score > 0.5) {
                errorCategories.incorrect_diagnosis.count++;
                errorCategories.incorrect_diagnosis.examples.push(index);
            }
        });
        
        // Create table rows
        const tableBody = document.getElementById('error-analysis-table');
        tableBody.innerHTML = '';
        
        let hasErrors = false;
        Object.entries(errorCategories).forEach(([category, data]) => {
            if (data.count > 0) {
                hasErrors = true;
                const row = document.createElement('tr');
                
                // Format category name for display
                const displayName = category.split('_').map(word => 
                    word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' ');
                
                // Calculate percentage
                const percentage = ((data.count / examples.length) * 100).toFixed(1);
                
                // Format example links
                const exampleLinks = data.examples.map(index => 
                    `<a href="#" class="example-link" data-example-index="${index}">Ex ${index+1}</a>`
                ).join(', ');
                
                row.innerHTML = `
                    <td>${displayName}</td>
                    <td>${data.count}</td>
                    <td>${percentage}%</td>
                    <td>${exampleLinks}</td>
                `;
                
                tableBody.appendChild(row);
            }
        });
        
        // If no errors found, show a message
        if (!hasErrors) {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="4" class="text-center text-success">No significant error patterns detected!</td>`;
            tableBody.appendChild(row);
        }
        
        // Add click handlers for example links
        document.querySelectorAll('.example-link').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const index = parseInt(this.dataset.exampleIndex);
                
                // Switch to examples tab and expand the selected example
                document.getElementById('examples-tab').click();
                
                // Ensure the example is visible (change filter if needed)
                document.getElementById('example-filter').value = 'all';
                document.querySelectorAll('.accordion-item').forEach(item => {
                    item.style.display = 'block';
                });
                
                // Expand the accordion for this example
                const targetAccordion = document.getElementById(`example-${index}`);
                const bsCollapse = new bootstrap.Collapse(targetAccordion, { toggle: true });
                
                // Scroll to the example
                targetAccordion.scrollIntoView({ behavior: 'smooth', block: 'center' });
            });
        });
    }
    
    // Helper function to truncate text
    function truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
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
        const batchSize = parseInt(document.getElementById('batch-size').value);
        log(`Batch size: ${batchSize === 0 ? 'All examples' : batchSize}`);
        
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
                batch_size: batchSize,
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
                
                // Add button to view detailed results
                const viewResultsBtn = document.createElement('button');
                viewResultsBtn.className = 'btn btn-info btn-sm mt-2 mb-2';
                viewResultsBtn.innerHTML = '<i class="fa-solid fa-chart-line me-1"></i> View Detailed Results';
                viewResultsBtn.addEventListener('click', function() {
                    // Prepare the model results data with robust error handling
                    const modelResultsData = {
                        experiment_id: data.experiment_id,
                        iteration: data.iterations || 0,
                        batch_size: batchSize,
                        model: 'gemini-1.5-flash',
                        optimizer_strategy: optimizerStrategy || 'Unknown strategy',
                        // Handle missing examples safely
                        examples: (data.initial && Array.isArray(data.initial.examples)) ? 
                                  data.initial.examples : []
                    };
                    
                    // Add metrics depending on whether optimization was performed
                    if (data.optimized && data.optimized.metrics) {
                        modelResultsData.metrics = data.optimized.metrics;
                    } else if (data.initial && data.initial.metrics) {
                        modelResultsData.metrics = data.initial.metrics;
                    }
                    
                    // Show the detailed results modal
                    showModelResults(modelResultsData);
                });
                
                // Add the button to the logs area
                const btnContainer = document.createElement('div');
                btnContainer.className = 'text-center';
                btnContainer.appendChild(viewResultsBtn);
                trainingLogsEl.appendChild(btnContainer);
                
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
                    
                    // Update metrics display if we have the data
                    if (data.initial && data.initial.metrics) {
                        updateMetricsDisplay(data.initial.metrics);
                    } else if (data.best_score) {
                        // Create a minimal metrics object if we don't have proper metrics
                        const basicMetrics = {
                            avg_score: data.best_score,
                            perfect_matches: 0,
                            total_examples: 1,
                            perfect_match_percent: 0
                        };
                        updateMetricsDisplay(basicMetrics);
                    }
                    
                    // Make sure we have valid metrics and system/output prompts
                    if (data.initial && data.initial.metrics) {
                        // Add to training history for chart with proper data
                        trainingHistory.push({
                            iteration: data.initial_iteration || 0,
                            metrics: data.initial.metrics,
                            system_prompt: data.initial.system_prompt || system_prompt,
                            output_prompt: data.initial.output_prompt || output_prompt
                        });
                    } else {
                        // Fallback - create a basic entry with the input prompts if data structure is unexpected
                        console.warn('Unexpected data structure in response:', data);
                        log('Warning: Received unexpected data structure from server');
                        
                        // Use whatever metrics we have available or create a placeholder
                        const fallbackMetrics = {
                            avg_score: data.best_score || 0,
                            perfect_matches: 0,
                            total_examples: 1,
                            perfect_match_percent: 0
                        };
                        
                        trainingHistory.push({
                            iteration: 0,
                            metrics: fallbackMetrics,
                            system_prompt: system_prompt,
                            output_prompt: output_prompt
                        });
                        
                        // Just update the metrics display with our fallback data
                        updateMetricsDisplay(fallbackMetrics);
                    }
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
        // Validate input objects and their properties
        if (!before || !after) {
            console.warn('Missing before/after data in optimization comparison');
            return;
        }
        
        // Make sure metrics exist and have required properties
        const beforeMetrics = before.metrics || { avg_score: 0, perfect_matches: 0, total_examples: 0 };
        const afterMetrics = after.metrics || { avg_score: 0, perfect_matches: 0, total_examples: 0 };
        
        optimizationStatusEl.style.display = 'none';
        optimizationResultsEl.style.display = 'block';
        
        // Safely display scores with fallbacks
        beforeScoreEl.textContent = ((beforeMetrics.avg_score || 0) * 100).toFixed(1) + '%';
        afterScoreEl.textContent = ((afterMetrics.avg_score || 0) * 100).toFixed(1) + '%';
        
        // Show reasoning if available
        if (after.reasoning) {
            optimizerReasoningEl.innerHTML = `<p>${after.reasoning.replace(/\n/g, '<br>')}</p>`;
        } else {
            optimizerReasoningEl.innerHTML = '<p class="text-muted">No reasoning available</p>';
        }
        
        // Add classes for improvement visualization
        if ((afterMetrics.avg_score || 0) > (beforeMetrics.avg_score || 0)) {
            afterScoreEl.classList.add('text-success');
        } else if ((afterMetrics.avg_score || 0) < (beforeMetrics.avg_score || 0)) {
            afterScoreEl.classList.add('text-danger');
        }
        
        // Store the optimized prompts for comparison with fallbacks
        currentComparisonData = {
            original: {
                system_prompt: before.system_prompt || 'No system prompt available',
                output_prompt: before.output_prompt || 'No output prompt available'
            },
            optimized: {
                system_prompt: after.system_prompt || 'No system prompt available',
                output_prompt: after.output_prompt || 'No output prompt available',
                reasoning: after.reasoning || ''
            }
        };
    }
    
    // Show prompt comparison modal
    function showPromptComparison() {
        if (!currentComparisonData) {
            showAlert('No optimization data available for comparison', 'warning');
            return;
        }
        
        // Populate comparison modal
        document.getElementById('original-system-prompt').textContent = currentComparisonData.original.system_prompt;
        document.getElementById('optimized-system-prompt').textContent = currentComparisonData.optimized.system_prompt;
        document.getElementById('original-output-prompt').textContent = currentComparisonData.original.output_prompt;
        document.getElementById('optimized-output-prompt').textContent = currentComparisonData.optimized.output_prompt;
        
        // Generate changes summary
        const changesList = document.getElementById('changes-summary');
        const reasoning = currentComparisonData.optimized.reasoning;
        
        if (reasoning) {
            // Look for key improvements in the reasoning text
            const changesHTML = generateChangesSummary(reasoning);
            changesList.innerHTML = changesHTML;
        } else {
            changesList.innerHTML = '<li class="text-muted">No detailed changes available</li>';
        }
        
        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById('prompt-comparison-modal'));
        modal.show();
    }
    
    // Generate summary of changes from optimizer reasoning
    function generateChangesSummary(reasoning) {
        // Split reasoning into paragraphs and look for key points
        const paragraphs = reasoning.split('\n').filter(p => p.trim().length > 0);
        let changesFound = [];
        
        // Keywords to look for that might indicate a change
        const changeKeywords = [
            'add', 'added', 'adding', 
            'remov', 'removed', 'removing',
            'chang', 'changed', 'changing',
            'modify', 'modified', 'modifying',
            'enhanc', 'enhanced', 'enhancing',
            'improv', 'improved', 'improving',
            'replac', 'replaced', 'replacing',
            'refin', 'refined', 'refining'
        ];
        
        // Look for sentences containing change keywords
        paragraphs.forEach(paragraph => {
            const sentences = paragraph.split(/[.!?]+/).filter(s => s.trim().length > 0);
            
            sentences.forEach(sentence => {
                const sentenceLower = sentence.toLowerCase().trim();
                
                // Check if the sentence contains any change keywords
                if (changeKeywords.some(keyword => sentenceLower.includes(keyword))) {
                    changesFound.push(sentence.trim());
                }
            });
        });
        
        // If we found specific changes, display them
        if (changesFound.length > 0) {
            // Limit to 5 most significant changes
            changesFound = changesFound.slice(0, 5);
            return changesFound.map(change => `<li>${change}.</li>`).join('');
        }
        
        // Fallback: just use the first few sentences from reasoning
        const firstFewSentences = reasoning.split(/[.!?]+/).filter(s => s.trim().length > 0).slice(0, 3);
        return firstFewSentences.map(sentence => `<li>${sentence.trim()}.</li>`).join('');
    }
    
    // Apply optimized prompts to the editor
    function applyOptimizedPrompts() {
        if (!currentComparisonData || !currentComparisonData.optimized) {
            showAlert('No optimized prompts available to apply', 'warning');
            return;
        }
        
        // Apply the optimized prompts to the editor fields
        systemPromptEl.value = currentComparisonData.optimized.system_prompt;
        outputPromptEl.value = currentComparisonData.optimized.output_prompt;
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('prompt-comparison-modal'));
        modal.hide();
        
        // Show success message
        showAlert('Optimized prompts applied successfully', 'success');
    }
    
    // Toggle between accordion and comparison views
    function toggleExampleView(viewType) {
        const accordionView = document.getElementById('accordion-view');
        const comparisonView = document.getElementById('comparison-view');
        const accordionBtn = document.getElementById('view-accordion');
        const comparisonBtn = document.getElementById('view-comparison');
        
        if (viewType === 'accordion') {
            accordionView.style.display = 'block';
            comparisonView.style.display = 'none';
            
            accordionBtn.classList.add('active');
            comparisonBtn.classList.remove('active');
        } else {
            accordionView.style.display = 'none';
            comparisonView.style.display = 'block';
            
            accordionBtn.classList.remove('active');
            comparisonBtn.classList.add('active');
            
            // Update the comparison view in case it's the first time viewing
            updateComparisonView();
        }
    }
    
    // Global variables to store example data for navigation
    let currentExamples = [];
    let currentExampleIndex = 0;
    
    // Update the comparison view with current example
    function updateComparisonView() {
        if (!currentExamples || currentExamples.length === 0) {
            document.getElementById('comparison-view').innerHTML = '<div class="alert alert-info m-3">No examples available for comparison</div>';
            return;
        }
        
        const example = currentExamples[currentExampleIndex];
        if (!example) return;
        
        // Update example counter
        document.getElementById('current-example-num').textContent = currentExampleIndex + 1;
        document.getElementById('total-examples').textContent = currentExamples.length;
        
        // Fill in example details
        document.getElementById('comparison-user-input').textContent = example.user_input || '';
        document.getElementById('comparison-ground-truth').textContent = example.ground_truth_output || '';
        
        // For initial vs optimized responses, we need to determine what data we have
        let initialResponse = '';
        let optimizedResponse = '';
        let initialScore = 0;
        let optimizedScore = 0;
        
        // If we have separate initial and optimized responses
        if (example.initial_response && example.optimized_response) {
            initialResponse = example.initial_response;
            optimizedResponse = example.optimized_response;
            initialScore = example.initial_score || 0;
            optimizedScore = example.optimized_score || example.score || 0;
        } 
        // If we only have model_response (single response)
        else if (example.model_response) {
            initialResponse = example.model_response;
            optimizedResponse = example.model_response;
            initialScore = example.score || 0;
            optimizedScore = example.score || 0;
        }
        
        // Set the response elements
        document.getElementById('comparison-initial-response').textContent = initialResponse;
        document.getElementById('comparison-optimized-response').textContent = optimizedResponse;
        
        // Update score badges
        document.getElementById('initial-score-badge').textContent = `Score: ${(initialScore * 100).toFixed(1)}%`;
        document.getElementById('optimized-score-badge').textContent = `Score: ${(optimizedScore * 100).toFixed(1)}%`;
        
        // Update improvement text
        const improvementEl = document.getElementById('improvement-text');
        if (optimizedScore > initialScore) {
            const improvement = ((optimizedScore - initialScore) * 100).toFixed(1);
            improvementEl.textContent = `Response improved by ${improvement}% after prompt optimization`;
            document.getElementById('comparison-improvement').className = 'alert alert-success';
        } else if (optimizedScore < initialScore) {
            const decrease = ((initialScore - optimizedScore) * 100).toFixed(1);
            improvementEl.textContent = `Response decreased by ${decrease}% after prompt optimization`;
            document.getElementById('comparison-improvement').className = 'alert alert-danger';
        } else {
            improvementEl.textContent = 'No change in score between initial and optimized responses';
            document.getElementById('comparison-improvement').className = 'alert alert-info';
        }
        
        // Enable/disable navigation buttons
        document.getElementById('prev-example').disabled = currentExampleIndex === 0;
        document.getElementById('next-example').disabled = currentExampleIndex === currentExamples.length - 1;
    }
    
    // Navigate between examples in comparison view
    function navigateExample(direction) {
        const newIndex = currentExampleIndex + direction;
        
        if (newIndex >= 0 && newIndex < currentExamples.length) {
            currentExampleIndex = newIndex;
            updateComparisonView();
        }
    }
    
    // Show the most recent comparison results in a dedicated dialog
    function showLatestComparisons() {
        // Check if we have any examples to show
        if (!currentExamples || currentExamples.length === 0) {
            // If no training has been done yet, show a message
            showAlert('No examples available yet. Run training first to see comparisons.', 'warning');
            return;
        }
        
        try {
            // Prepare the model results modal with the current examples
            // Limit to 10 examples max to avoid memory issues
            const limitedExamples = currentExamples.slice(0, 10);
            
            const modelResultsData = {
                experiment_id: currentExperimentId || 'current_session',
                iteration: 'Latest',
                batch_size: 'Up to 10 examples',
                model: 'gemini-1.5-flash',
                optimizer_strategy: document.getElementById('optimizer-strategy').value,
                examples: limitedExamples,
                metrics: {
                    avg_score: parseFloat(document.getElementById('current-score').textContent) / 100 || 0,
                    perfect_matches: (document.getElementById('perfect-matches').textContent || '0/0').split('/')[0],
                    total_examples: (document.getElementById('perfect-matches').textContent || '0/0').split('/')[1] || 0
                }
            };
            
            // Show the model results modal
            showModelResults(modelResultsData);
            
            // Automatically switch to the Comparisons tab
            setTimeout(() => {
                document.querySelector('#comparisons-tab').click();
                updateComparisonTabView();
            }, 300);
        } catch (error) {
            console.error("Error in showing comparisons:", error);
            showAlert('There was an error showing the comparisons. Please check the console for details.', 'danger');
        }
    }
    
    // Update the comparison tab view
    function updateComparisonTabView() {
        if (!currentExamples || currentExamples.length === 0) {
            document.getElementById('comparison-content').style.display = 'none';
            document.getElementById('no-comparison-examples').style.display = 'block';
            return;
        }
        
        document.getElementById('comparison-content').style.display = 'block';
        document.getElementById('no-comparison-examples').style.display = 'none';
        
        // Update the navigation counter
        document.getElementById('comparison-example-counter').textContent = 
            `Example ${currentExampleIndex + 1} of ${currentExamples.length}`;
        
        // Disable/enable navigation buttons as needed
        document.getElementById('comparison-prev-example').disabled = currentExampleIndex === 0;
        document.getElementById('comparison-next-example').disabled = currentExampleIndex === currentExamples.length - 1;
        
        // Get the current example
        const example = currentExamples[currentExampleIndex];
        
        // Update the comparison view content
        document.getElementById('comparison-user-input').textContent = example.user_input || '';
        document.getElementById('comparison-ground-truth').textContent = example.ground_truth_output || '';
        document.getElementById('comparison-model-response').textContent = example.model_response || '';
        document.getElementById('comparison-improved-response').textContent = example.improved_response || example.model_response || '';
        
        // Update scores
        const initialScore = example.score ? (example.score * 100).toFixed(1) : '0.0';
        document.getElementById('comparison-initial-score').textContent = `Score: ${initialScore}%`;
        
        // If there's an improved response with a score
        let improvedScore = initialScore;
        if (example.improved_score) {
            improvedScore = (example.improved_score * 100).toFixed(1);
        }
        document.getElementById('comparison-improved-score').textContent = `Score: ${improvedScore}%`;
        
        // Highlight score improvement
        if (parseFloat(improvedScore) > parseFloat(initialScore)) {
            document.getElementById('comparison-improved-score').classList.remove('bg-secondary');
            document.getElementById('comparison-improved-score').classList.add('bg-success');
        } else {
            document.getElementById('comparison-improved-score').classList.remove('bg-success');
            document.getElementById('comparison-improved-score').classList.add('bg-secondary');
        }
    }
    
    // Show spinner with safety timeout
    function showSpinner() {
        spinner.style.display = 'flex';
        
        // Set a safety timeout to hide spinner if it gets stuck
        if (spinnerTimeout) {
            clearTimeout(spinnerTimeout);
        }
        
        // Automatically hide spinner after 30 seconds if it gets stuck
        spinnerTimeout = setTimeout(() => {
            hideSpinner();
            showAlert('Operation timed out. If you were loading data, please try again or check the logs for errors.', 'warning');
            console.warn('Spinner timeout triggered - operation took too long');
        }, 30000);
    }
    
    // Hide spinner and clear timeout
    function hideSpinner() {
        spinner.style.display = 'none';
        
        // Clear the safety timeout
        if (spinnerTimeout) {
            clearTimeout(spinnerTimeout);
            spinnerTimeout = null;
        }
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
                    
                    // Add button to view detailed results of best version
                    resultsHtml += `<div class="text-center mt-3 mb-2">
                        <button id="view-validation-results-btn" class="btn btn-info">
                            <i class="fa-solid fa-magnifying-glass-chart me-1"></i> View Detailed Results
                        </button>
                    </div>`;
                }
                
                resultsHtml += '</div>';
                
                // Add to the logs area
                document.querySelector('.validation-container').innerHTML = resultsHtml;
                
                // Add event listener to view detailed results button
                const viewValidationResultsBtn = document.getElementById('view-validation-results-btn');
                if (viewValidationResultsBtn && bestVersion) {
                    viewValidationResultsBtn.addEventListener('click', function() {
                        // Prepare detailed results data for the best version
                        const bestResults = data.validation_results[bestVersion];
                        const modelResultsData = {
                            experiment_id: currentExperimentId || 'validation',
                            iteration: bestVersion,
                            batch_size: 'All validation examples',
                            model: config.model || 'gemini-1.5-flash',
                            optimizer_strategy: 'Validation run',
                            metrics: bestResults.metrics,
                            examples: bestResults.examples || []
                        };
                        
                        // Show the detailed results modal
                        showModelResults(modelResultsData);
                    });
                }
                
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