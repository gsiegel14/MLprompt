/**
 * Five-API Workflow JavaScript
 * 
 * This script handles the frontend interaction for the 5-API call workflow:
 * 1. Google Vertex API #1: Primary LLM inference
 * 2. Hugging Face API: First external validation
 * 3. Google Vertex API #2: Optimizer LLM for prompt refinement
 * 4. Google Vertex API #3: Optimizer LLM reruns on original dataset
 * 5. Hugging Face API: Second external validation on refined outputs
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI components
    initializeForm();
    setupEventListeners();
    updateMetricsUI();
});

/**
 * Initialize the UI form elements
 */
function initializeForm() {
    // Set default metrics
    const defaultMetrics = ['exact_match', 'bleu'];
    const metricsContainer = document.getElementById('hf-metrics-container');
    
    if (metricsContainer) {
        defaultMetrics.forEach(metric => {
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'hf-metrics';
            checkbox.value = metric;
            checkbox.id = `metric-${metric}`;
            checkbox.checked = true;
            
            const label = document.createElement('label');
            label.htmlFor = `metric-${metric}`;
            label.textContent = metric;
            
            const div = document.createElement('div');
            div.className = 'form-check';
            div.appendChild(checkbox);
            div.appendChild(label);
            
            metricsContainer.appendChild(div);
        });
        
        // Add additional metrics
        ['rouge', 'bertscore', 'f1'].forEach(metric => {
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'hf-metrics';
            checkbox.value = metric;
            checkbox.id = `metric-${metric}`;
            
            const label = document.createElement('label');
            label.htmlFor = `metric-${metric}`;
            label.textContent = metric;
            
            const div = document.createElement('div');
            div.className = 'form-check';
            div.appendChild(checkbox);
            div.appendChild(label);
            
            metricsContainer.appendChild(div);
        });
    }
    
    // Load optimization strategies
    loadOptimizationStrategies();
}

/**
 * Set up event listeners for UI elements
 */
function setupEventListeners() {
    const runWorkflowButton = document.getElementById('run-workflow-button');
    if (runWorkflowButton) {
        runWorkflowButton.addEventListener('click', runFiveApiWorkflow);
    }
    
    // Copy buttons for prompts
    const copyButtons = document.querySelectorAll('.copy-button');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                copyToClipboard(targetElement.value);
                showCopyFeedback(this);
            }
        });
    });
    
    // Reset metrics button
    const resetButton = document.getElementById('reset-metrics-button');
    if (resetButton) {
        resetButton.addEventListener('click', resetMetrics);
    }
}

/**
 * Load optimization strategies from the server
 */
function loadOptimizationStrategies() {
    fetch('/get_optimization_strategies')
        .then(response => response.json())
        .then(data => {
            const strategiesSelect = document.getElementById('optimizer-strategy');
            if (strategiesSelect && data.strategies) {
                // Clear existing options
                strategiesSelect.innerHTML = '';
                
                // Add each strategy as an option
                data.strategies.forEach(strategy => {
                    const option = document.createElement('option');
                    option.value = strategy;
                    option.textContent = strategy.replace(/_/g, ' ');
                    strategiesSelect.appendChild(option);
                });
                
                // Select the default "reasoning_first" strategy
                const defaultStrategy = 'reasoning_first';
                for (let i = 0; i < strategiesSelect.options.length; i++) {
                    if (strategiesSelect.options[i].value === defaultStrategy) {
                        strategiesSelect.selectedIndex = i;
                        break;
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error loading optimization strategies:', error);
            showAlert('error', 'Failed to load optimization strategies');
        });
}

/**
 * Run the 5-API workflow
 */
function runFiveApiWorkflow() {
    // Show loading UI
    const runButton = document.getElementById('run-workflow-button');
    const originalButtonText = runButton.innerHTML;
    runButton.disabled = true;
    runButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...';
    
    // Clear results
    document.getElementById('workflow-results').style.display = 'none';
    
    // Get form data
    const systemPrompt = document.getElementById('system-prompt').value;
    const outputPrompt = document.getElementById('output-prompt').value;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 10;
    const optimizerStrategy = document.getElementById('optimizer-strategy').value;
    
    // Get selected metrics
    const metricCheckboxes = document.querySelectorAll('input[name="hf-metrics"]:checked');
    const selectedMetrics = Array.from(metricCheckboxes).map(cb => cb.value);
    
    // Validate inputs
    if (!systemPrompt || !outputPrompt) {
        showAlert('error', 'System prompt and output prompt are required');
        runButton.disabled = false;
        runButton.innerHTML = originalButtonText;
        return;
    }
    
    // Prepare request data
    const requestData = {
        system_prompt: systemPrompt,
        output_prompt: outputPrompt,
        batch_size: batchSize,
        optimizer_strategy: optimizerStrategy,
        hf_metrics: selectedMetrics
    };
    
    // Send request to server
    fetch('/five_api_workflow', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        // Restore button state
        runButton.disabled = false;
        runButton.innerHTML = originalButtonText;
        
        if (data.error) {
            showAlert('error', `Workflow failed: ${data.error}`);
            return;
        }
        
        // Show results
        displayWorkflowResults(data);
        
        // Store metrics for comparisons
        storeMetrics(data);
        
        // Update charts
        updateMetricsUI();
        
        // Show success message
        showAlert('success', 'Workflow completed successfully!');
    })
    .catch(error => {
        console.error('Error running workflow:', error);
        showAlert('error', `Error running workflow: ${error.message}`);
        
        // Restore button state
        runButton.disabled = false;
        runButton.innerHTML = originalButtonText;
    });
}

/**
 * Display the workflow results in the UI
 */
function displayWorkflowResults(data) {
    const resultsContainer = document.getElementById('workflow-results');
    resultsContainer.style.display = 'block';
    
    // Experiment ID
    document.getElementById('experiment-id').textContent = data.experiment_id;
    
    // Counts
    document.getElementById('examples-count').textContent = data.examples_count;
    document.getElementById('validation-count').textContent = data.validation_count;
    
    // Original Prompts
    document.getElementById('original-system-prompt').value = data.prompts.original.system_prompt;
    document.getElementById('original-output-prompt').value = data.prompts.original.output_prompt;
    
    // Optimized Prompts
    document.getElementById('optimized-system-prompt').value = data.prompts.optimized.system_prompt;
    document.getElementById('optimized-output-prompt').value = data.prompts.optimized.output_prompt;
    
    // Internal Metrics
    const internalMetricsElement = document.getElementById('internal-metrics');
    internalMetricsElement.innerHTML = '';
    
    const internalMetrics = data.metrics.internal;
    for (const key in internalMetrics) {
        const value = internalMetrics[key];
        const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
        
        const row = document.createElement('tr');
        
        const keyCell = document.createElement('td');
        keyCell.textContent = key;
        row.appendChild(keyCell);
        
        const valueCell = document.createElement('td');
        valueCell.textContent = formattedValue;
        row.appendChild(valueCell);
        
        internalMetricsElement.appendChild(row);
    }
    
    // Hugging Face Metrics
    const hfMetricsElement = document.getElementById('hf-metrics');
    hfMetricsElement.innerHTML = '';
    
    // Create a row for each metric, with columns for original and optimized
    const originalMetrics = data.metrics.huggingface.original;
    const optimizedMetrics = data.metrics.huggingface.optimized;
    
    // Combine all metric keys
    const allMetricKeys = new Set([
        ...Object.keys(originalMetrics),
        ...Object.keys(optimizedMetrics)
    ]);
    
    allMetricKeys.forEach(key => {
        const originalValue = originalMetrics[key];
        const optimizedValue = optimizedMetrics[key];
        
        const formattedOriginal = typeof originalValue === 'number' ? originalValue.toFixed(4) : originalValue || 'N/A';
        const formattedOptimized = typeof optimizedValue === 'number' ? optimizedValue.toFixed(4) : optimizedValue || 'N/A';
        
        // Determine if the optimized value is better
        let improvement = '';
        if (typeof originalValue === 'number' && typeof optimizedValue === 'number') {
            const diff = optimizedValue - originalValue;
            if (diff > 0) {
                improvement = `<span class="text-success">+${diff.toFixed(4)}</span>`;
            } else if (diff < 0) {
                improvement = `<span class="text-danger">${diff.toFixed(4)}</span>`;
            } else {
                improvement = '<span class="text-secondary">0</span>';
            }
        }
        
        const row = document.createElement('tr');
        
        const keyCell = document.createElement('td');
        keyCell.textContent = key;
        row.appendChild(keyCell);
        
        const originalCell = document.createElement('td');
        originalCell.textContent = formattedOriginal;
        row.appendChild(originalCell);
        
        const optimizedCell = document.createElement('td');
        optimizedCell.textContent = formattedOptimized;
        row.appendChild(optimizedCell);
        
        const improvementCell = document.createElement('td');
        improvementCell.innerHTML = improvement;
        row.appendChild(improvementCell);
        
        hfMetricsElement.appendChild(row);
    });
}

/**
 * Store metrics data for historical comparison
 */
function storeMetrics(data) {
    let metricsHistory = JSON.parse(localStorage.getItem('metricsHistory') || '[]');
    
    // Add timestamp and extract key metrics
    const timestamp = new Date().toISOString();
    const newEntry = {
        timestamp: timestamp,
        experiment_id: data.experiment_id,
        internal: {
            avg_score: data.metrics.internal.avg_score || 0,
            perfect_match_percent: data.metrics.internal.perfect_match_percent || 0
        },
        huggingface: {
            original: {},
            optimized: {}
        }
    };
    
    // Extract all Hugging Face metrics
    const originalMetrics = data.metrics.huggingface.original;
    const optimizedMetrics = data.metrics.huggingface.optimized;
    
    for (const key in originalMetrics) {
        if (typeof originalMetrics[key] === 'number') {
            newEntry.huggingface.original[key] = originalMetrics[key];
        }
    }
    
    for (const key in optimizedMetrics) {
        if (typeof optimizedMetrics[key] === 'number') {
            newEntry.huggingface.optimized[key] = optimizedMetrics[key];
        }
    }
    
    // Add to history (limit to 10 entries)
    metricsHistory.push(newEntry);
    if (metricsHistory.length > 10) {
        metricsHistory = metricsHistory.slice(-10);
    }
    
    // Save to localStorage
    localStorage.setItem('metricsHistory', JSON.stringify(metricsHistory));
}

/**
 * Update the metrics UI with charts
 */
function updateMetricsUI() {
    const metricsHistory = JSON.parse(localStorage.getItem('metricsHistory') || '[]');
    
    // If no metrics, hide the charts
    if (metricsHistory.length === 0) {
        document.getElementById('metrics-charts-container').style.display = 'none';
        return;
    }
    
    // Show the charts container
    document.getElementById('metrics-charts-container').style.display = 'block';
    
    // Get metrics for charts
    const labels = metricsHistory.map((_, i) => `Run ${i + 1}`);
    
    const internalScores = metricsHistory.map(entry => entry.internal.avg_score);
    const perfectMatch = metricsHistory.map(entry => entry.internal.perfect_match_percent);
    
    // Extract HF metrics if available (focus on exact_match and bleu)
    const originalExactMatch = metricsHistory.map(entry => 
        entry.huggingface?.original?.exact_match || null);
    const optimizedExactMatch = metricsHistory.map(entry => 
        entry.huggingface?.optimized?.exact_match || null);
        
    const originalBleu = metricsHistory.map(entry => 
        entry.huggingface?.original?.bleu || null);
    const optimizedBleu = metricsHistory.map(entry => 
        entry.huggingface?.optimized?.bleu || null);
    
    // Draw charts
    drawPerformanceChart('internal-metrics-chart', labels, 
        [internalScores, perfectMatch],
        ['Average Score', 'Perfect Match %'],
        ['rgba(54, 162, 235, 0.8)', 'rgba(75, 192, 192, 0.8)']
    );
    
    drawComparisonChart('hf-exact-match-chart', labels, 
        [originalExactMatch, optimizedExactMatch],
        ['Original Exact Match', 'Optimized Exact Match'],
        ['rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)']
    );
    
    drawComparisonChart('hf-bleu-chart', labels, 
        [originalBleu, optimizedBleu],
        ['Original BLEU', 'Optimized BLEU'],
        ['rgba(255, 159, 64, 0.8)', 'rgba(75, 192, 192, 0.8)']
    );
}

/**
 * Draw a line chart for performance metrics
 */
function drawPerformanceChart(canvasId, labels, datasets, dataLabels, colors) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.metricsCharts && window.metricsCharts[canvasId]) {
        window.metricsCharts[canvasId].destroy();
    }
    
    // Initialize charts object if it doesn't exist
    if (!window.metricsCharts) {
        window.metricsCharts = {};
    }
    
    // Create datasets for Chart.js
    const chartDatasets = datasets.map((data, i) => ({
        label: dataLabels[i],
        data: data,
        borderColor: colors[i],
        backgroundColor: colors[i].replace('0.8', '0.2'),
        tension: 0.3,
        fill: false
    }));
    
    // Create chart
    window.metricsCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: chartDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            }
        }
    });
}

/**
 * Draw a comparison chart for original vs optimized metrics
 */
function drawComparisonChart(canvasId, labels, datasets, dataLabels, colors) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.metricsCharts && window.metricsCharts[canvasId]) {
        window.metricsCharts[canvasId].destroy();
    }
    
    // Initialize charts object if it doesn't exist
    if (!window.metricsCharts) {
        window.metricsCharts = {};
    }
    
    // Create datasets for Chart.js
    const chartDatasets = datasets.map((data, i) => ({
        label: dataLabels[i],
        data: data,
        borderColor: colors[i],
        backgroundColor: colors[i].replace('0.8', '0.2'),
        tension: 0.3,
        fill: false
    }));
    
    // Create chart
    window.metricsCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: chartDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            }
        }
    });
}

/**
 * Reset metrics history
 */
function resetMetrics() {
    if (confirm('Are you sure you want to reset all metrics history?')) {
        localStorage.removeItem('metricsHistory');
        updateMetricsUI();
        showAlert('info', 'Metrics history has been reset');
    }
}

/**
 * Show an alert message
 */
function showAlert(type, message) {
    const alertContainer = document.getElementById('alert-container');
    if (!alertContainer) return;
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const bsAlert = new bootstrap.Alert(alertDiv);
        bsAlert.close();
    }, 5000);
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
}

/**
 * Show feedback when copying to clipboard
 */
function showCopyFeedback(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="bi bi-check"></i> Copied!';
    
    setTimeout(() => {
        button.innerHTML = originalText;
    }, 2000);
}
/**
 * Five-step API Workflow Visualization
 * 
 * This script handles the UI for the 5-step ML prompt optimization workflow:
 * 1. Primary LLM Inference
 * 2. Hugging Face Evaluation
 * 3. Optimizer LLM
 * 4. Refined LLM Inference
 * 5. Second Evaluation
 */

// Initialize workflow state
let workflowState = {
  currentStep: 0,
  flowId: null,
  promptId: null,
  datasetId: null,
  metrics: {},
  iterations: [],
  currentIteration: 0,
  maxIterations: 5,
  isRunning: false
};

// DOM references
const workflowSteps = document.querySelectorAll('.workflow-step');
const startWorkflowBtn = document.getElementById('start-workflow');
const stopWorkflowBtn = document.getElementById('stop-workflow');
const progressBar = document.getElementById('workflow-progress');
const statusMessage = document.getElementById('status-message');
const promptSelector = document.getElementById('prompt-selector');
const datasetSelector = document.getElementById('dataset-selector');
const metricsContainer = document.getElementById('metrics-container');
const promptComparison = document.getElementById('prompt-comparison');

// Chart for metrics visualization
let metricsChart = null;

// Initialize workflow UI
function initWorkflow() {
  // Set up event listeners
  startWorkflowBtn.addEventListener('click', startWorkflow);
  stopWorkflowBtn.addEventListener('click', stopWorkflow);
  
  // Load prompts and datasets
  loadPrompts();
  loadDatasets();
  
  // Initialize metrics chart
  initMetricsChart();
  
  // Update UI state
  updateUIState();
}

// Load available prompts from API
async function loadPrompts() {
  try {
    const response = await fetch('/api/v1/prompts');
    if (!response.ok) throw new Error('Failed to load prompts');
    
    const prompts = await response.json();
    
    // Clear selector
    promptSelector.innerHTML = '<option value="">Select a prompt</option>';
    
    // Add options
    prompts.forEach(prompt => {
      const option = document.createElement('option');
      option.value = prompt.id;
      option.textContent = prompt.name || `Prompt ${prompt.id.substring(0, 8)}`;
      promptSelector.appendChild(option);
    });
  } catch (error) {
    console.error('Error loading prompts:', error);
    showError('Failed to load prompts');
  }
}

// Load available datasets from API
async function loadDatasets() {
  try {
    const response = await fetch('/api/v1/datasets');
    if (!response.ok) throw new Error('Failed to load datasets');
    
    const datasets = await response.json();
    
    // Clear selector
    datasetSelector.innerHTML = '<option value="">Select a dataset</option>';
    
    // Add options
    datasets.forEach(dataset => {
      const option = document.createElement('option');
      option.value = dataset.id;
      option.textContent = dataset.name || `Dataset ${dataset.id.substring(0, 8)}`;
      datasetSelector.appendChild(option);
    });
  } catch (error) {
    console.error('Error loading datasets:', error);
    showError('Failed to load datasets');
  }
}

// Initialize metrics chart
function initMetricsChart() {
  const ctx = document.getElementById('metrics-chart').getContext('2d');
  
  metricsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Baseline',
        data: [],
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1
      }, {
        label: 'Optimized',
        data: [],
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 1
        }
      },
      plugins: {
        tooltip: {
          mode: 'index',
          intersect: false
        },
        legend: {
          position: 'top'
        },
        title: {
          display: true,
          text: 'Prompt Performance Metrics'
        }
      }
    }
  });
}

// Start the workflow
async function startWorkflow() {
  // Validate inputs
  const promptId = promptSelector.value;
  const datasetId = datasetSelector.value;
  
  if (!promptId || !datasetId) {
    showError('Please select both a prompt and a dataset');
    return;
  }
  
  try {
    // Update UI
    workflowState.isRunning = true;
    workflowState.currentStep = 1;
    workflowState.promptId = promptId;
    workflowState.datasetId = datasetId;
    updateUIState();
    
    // Start optimization workflow
    const response = await fetch('/api/v1/optimization', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt_id: promptId,
        dataset_id: datasetId,
        target_metric: 'exact_match_score',
        max_iterations: workflowState.maxIterations
      })
    });
    
    if (!response.ok) throw new Error('Failed to start workflow');
    
    const data = await response.json();
    workflowState.flowId = data.flow_id;
    
    // Start polling for updates
    pollWorkflowStatus();
  } catch (error) {
    console.error('Error starting workflow:', error);
    showError('Failed to start workflow');
    workflowState.isRunning = false;
    updateUIState();
  }
}

// Stop the workflow
async function stopWorkflow() {
  if (!workflowState.flowId) return;
  
  try {
    const response = await fetch(`/api/v1/optimization/${workflowState.flowId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) throw new Error('Failed to stop workflow');
    
    // Update UI
    workflowState.isRunning = false;
    showMessage('Workflow stopped');
    updateUIState();
  } catch (error) {
    console.error('Error stopping workflow:', error);
    showError('Failed to stop workflow');
  }
}

// Poll for workflow status updates
async function pollWorkflowStatus() {
  if (!workflowState.isRunning || !workflowState.flowId) return;
  
  try {
    const response = await fetch(`/api/v1/optimization/${workflowState.flowId}`);
    if (!response.ok) throw new Error('Failed to get workflow status');
    
    const data = await response.json();
    
    // Update workflow state
    workflowState.currentIteration = data.current_iteration;
    workflowState.status = data.status;
    
    // Check if completed
    if (data.status === 'completed') {
      workflowState.isRunning = false;
      workflowState.currentStep = 5;
      showMessage('Workflow completed!');
      
      // Load results
      await loadWorkflowResults();
    } else if (data.status === 'failed') {
      workflowState.isRunning = false;
      showError('Workflow failed');
    } else if (data.status === 'cancelled') {
      workflowState.isRunning = false;
      showMessage('Workflow cancelled');
    } else {
      // Calculate current step based on iteration progress
      const stepProgress = (data.current_iteration / data.max_iterations) * 4;
      workflowState.currentStep = Math.min(Math.ceil(stepProgress) + 1, 5);
      
      // Continue polling
      setTimeout(pollWorkflowStatus, 2000);
    }
    
    // Update UI
    updateUIState();
  } catch (error) {
    console.error('Error polling workflow status:', error);
    showError('Failed to get workflow status');
    
    // Retry after a delay
    setTimeout(pollWorkflowStatus, 5000);
  }
}

// Load workflow results when completed
async function loadWorkflowResults() {
  try {
    const response = await fetch(`/api/v1/optimization/${workflowState.flowId}/results`);
    if (!response.ok) throw new Error('Failed to load workflow results');
    
    const results = await response.json();
    
    // Update state with results
    workflowState.iterations = results.iterations;
    workflowState.metrics = results.best_metrics;
    
    // Update UI with results
    updateMetricsChart();
    updatePromptComparison(results);
  } catch (error) {
    console.error('Error loading workflow results:', error);
    showError('Failed to load workflow results');
  }
}

// Update metrics chart with iteration data
function updateMetricsChart() {
  if (!workflowState.iterations || !workflowState.iterations.length) return;
  
  // Extract data for chart
  const labels = workflowState.iterations.map(iter => `Iteration ${iter.iteration}`);
  const baselineData = workflowState.iterations.map(iter => iter.baseline.metrics.exact_match_score || 0);
  const optimizedData = workflowState.iterations.map(iter => iter.optimized.metrics.exact_match_score || 0);
  
  // Update chart
  metricsChart.data.labels = labels;
  metricsChart.data.datasets[0].data = baselineData;
  metricsChart.data.datasets[1].data = optimizedData;
  metricsChart.update();
  
  // Show metrics container
  metricsContainer.classList.remove('hidden');
}

// Update prompt comparison view
function updatePromptComparison(results) {
  if (!results || !results.best_state) return;
  
  // Get original and best prompts
  const originalPrompt = {
    system: results.original_state.system_prompt,
    output: results.original_state.output_prompt
  };
  
  const bestPrompt = {
    system: results.best_state.system_prompt,
    output: results.best_state.output_prompt
  };
  
  // Build HTML
  let html = `
    <h3>Prompt Comparison</h3>
    <div class="comparison-container">
      <div class="prompt-column">
        <h4>Original Prompt</h4>
        <div class="prompt-box">
          <h5>System Prompt</h5>
          <pre>${escapeHtml(originalPrompt.system)}</pre>
          <h5>Output Prompt</h5>
          <pre>${escapeHtml(originalPrompt.output)}</pre>
        </div>
      </div>
      <div class="prompt-column">
        <h4>Optimized Prompt</h4>
        <div class="prompt-box">
          <h5>System Prompt</h5>
          <pre>${escapeHtml(bestPrompt.system)}</pre>
          <h5>Output Prompt</h5>
          <pre>${escapeHtml(bestPrompt.output)}</pre>
        </div>
      </div>
    </div>
    <button id="save-optimized" class="btn btn-primary mt-3">Save Optimized Prompt</button>
  `;
  
  // Update DOM
  promptComparison.innerHTML = html;
  promptComparison.classList.remove('hidden');
  
  // Add event listener for save button
  document.getElementById('save-optimized').addEventListener('click', saveOptimizedPrompt);
}

// Save the optimized prompt
async function saveOptimizedPrompt() {
  if (!workflowState.flowId) return;
  
  try {
    const response = await fetch(`/api/v1/optimization/${workflowState.flowId}/save`, {
      method: 'POST'
    });
    
    if (!response.ok) throw new Error('Failed to save optimized prompt');
    
    showMessage('Optimized prompt saved successfully!');
  } catch (error) {
    console.error('Error saving optimized prompt:', error);
    showError('Failed to save optimized prompt');
  }
}

// Update UI state based on workflow state
function updateUIState() {
  // Update steps
  workflowSteps.forEach((step, index) => {
    step.classList.remove('active', 'completed');
    
    if (index + 1 < workflowState.currentStep) {
      step.classList.add('completed');
    } else if (index + 1 === workflowState.currentStep) {
      step.classList.add('active');
    }
  });
  
  // Update progress
  const progress = (workflowState.currentStep / 5) * 100;
  progressBar.style.width = `${progress}%`;
  
  // Update buttons
  startWorkflowBtn.disabled = workflowState.isRunning;
  stopWorkflowBtn.disabled = !workflowState.isRunning;
  
  // Update selectors
  promptSelector.disabled = workflowState.isRunning;
  datasetSelector.disabled = workflowState.isRunning;
  
  // Update status message
  if (workflowState.isRunning) {
    const stepNames = [
      'Starting',
      'Primary LLM Inference',
      'Evaluation',
      'Optimizer LLM',
      'Refined Inference',
      'Final Evaluation'
    ];
    
    statusMessage.textContent = `${stepNames[workflowState.currentStep]} (Iteration ${workflowState.currentIteration}/${workflowState.maxIterations})`;
  } else if (workflowState.currentStep === 5) {
    statusMessage.textContent = 'Workflow completed';
  } else {
    statusMessage.textContent = 'Ready to start workflow';
  }
}

// Show error message
function showError(message) {
  const alert = document.createElement('div');
  alert.className = 'alert alert-danger alert-dismissible fade show';
  alert.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  
  const container = document.querySelector('.alerts-container');
  container.appendChild(alert);
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    alert.classList.remove('show');
    setTimeout(() => alert.remove(), 150);
  }, 5000);
}

// Show success/info message
function showMessage(message) {
  const alert = document.createElement('div');
  alert.className = 'alert alert-success alert-dismissible fade show';
  alert.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  
  const container = document.querySelector('.alerts-container');
  container.appendChild(alert);
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    alert.classList.remove('show');
    setTimeout(() => alert.remove(), 150);
  }, 5000);
}

// Helper function to escape HTML
function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initWorkflow);
