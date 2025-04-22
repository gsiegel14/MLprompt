/**
 * Evaluation Page JavaScript
 * This script handles the prompt evaluation interface with the 4-API workflow
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
    const evaluationSystemPromptTextarea = document.getElementById('evaluation-system-prompt');
    const evaluationOutputPromptTextarea = document.getElementById('evaluation-output-prompt');
    const evalDatasetSelect = document.getElementById('eval-dataset');
    const evalBatchSizeInput = document.getElementById('eval-batch-size');
    const runEvaluationBtn = document.getElementById('run-evaluation-btn');
    const experimentSelect = document.getElementById('experiment-select');
    const iterationSelect = document.getElementById('iteration-select');
    const loadPromptBtn = document.getElementById('load-prompt-btn');
    const saveEvaluationBtn = document.getElementById('save-evaluation-btn');
    const optimizeFromEvaluationBtn = document.getElementById('optimize-from-evaluation-btn');
    const goToFinalBtn = document.getElementById('go-to-final-btn');
    const spinner = document.getElementById('spinner');
    const alertContainer = document.getElementById('alert-container');
    const resultsBody = document.getElementById('results-body');
    
    // Metrics display elements
    const avgScoreDisplay = document.getElementById('avg-score');
    const perfectMatchesDisplay = document.getElementById('perfect-matches');
    const totalExamplesDisplay = document.getElementById('total-examples');
    const matchRateDisplay = document.getElementById('match-rate');
    
    // Chart
    let metricsChart = null;

    // Load default evaluation prompts
    loadDefaultEvaluationPrompts();
    
    // Load experiments
    loadExperiments();

    // Event listeners
    evalDatasetSelect.addEventListener('change', handleDatasetChange);
    runEvaluationBtn.addEventListener('click', runEvaluation);
    experimentSelect.addEventListener('change', loadIterations);
    iterationSelect.addEventListener('change', enableLoadButton);
    loadPromptBtn.addEventListener('click', loadSelectedPrompt);
    saveEvaluationBtn.addEventListener('click', saveEvaluationResults);
    optimizeFromEvaluationBtn.addEventListener('click', optimizePromptsFromEvaluation);
    goToFinalBtn.addEventListener('click', goToFinalPrompts);

    /**
     * Load default evaluation system and output prompts
     */
    function loadDefaultEvaluationPrompts() {
        // Default evaluation system prompt
        const defaultEvalSystemPrompt = 
            "You are an expert evaluator for LLM responses. Your task is to compare the model's response to the ground truth answer and provide a detailed evaluation. " +
            "Consider the following criteria:\n" +
            "1. Accuracy - Is the factual content correct?\n" +
            "2. Completeness - Does it address all aspects of the ground truth?\n" +
            "3. Conciseness - Is it appropriately concise without unnecessary information?\n" +
            "4. Relevance - Does it directly address the user's input?\n\n" +
            "Provide a score between 0.0 (completely incorrect) and 1.0 (perfect match).";
            
        // Default evaluation output prompt
        const defaultEvalOutputPrompt = 
            "Compare the model's response to the ground truth answer and evaluate it using the criteria outlined.\n\n" +
            "Provide your evaluation in the following format:\n" +
            "{\n" +
            '  "score": <numerical_score_between_0_and_1>,\n' +
            '  "reasoning": "<brief explanation of your reasoning>",\n' +
            '  "strengths": ["<strength1>", "<strength2>", ...],\n' +
            '  "weaknesses": ["<weakness1>", "<weakness2>", ...]\n' +
            "}";
        
        // Set values
        evaluationSystemPromptTextarea.value = defaultEvalSystemPrompt;
        evaluationOutputPromptTextarea.value = defaultEvalOutputPrompt;
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
     * Load iterations for the selected experiment
     */
    function loadIterations() {
        const experimentId = experimentSelect.value;
        
        if (!experimentId) {
            iterationSelect.innerHTML = '<option value="" selected>Select iteration...</option>';
            iterationSelect.disabled = true;
            loadPromptBtn.disabled = true;
            return;
        }
        
        showSpinner();
        
        fetch(`/experiments/${experimentId}`)
            .then(response => response.json())
            .then(data => {
                hideSpinner();
                
                if (data.iterations && data.iterations.length > 0) {
                    iterationSelect.innerHTML = '<option value="" selected>Select iteration...</option>';
                    
                    // Add iteration options
                    data.iterations.forEach(iteration => {
                        const option = document.createElement('option');
                        option.value = iteration.iteration;
                        option.textContent = `Iteration ${iteration.iteration} (Score: ${iteration.avg_score.toFixed(2)})`;
                        iterationSelect.appendChild(option);
                    });
                    
                    iterationSelect.disabled = false;
                } else {
                    iterationSelect.innerHTML = '<option value="" selected>No iterations available</option>';
                    iterationSelect.disabled = true;
                    loadPromptBtn.disabled = true;
                    showAlert('No iterations found for this experiment.', 'warning');
                }
            })
            .catch(error => {
                hideSpinner();
                showAlert(`Error loading iterations: ${error}`, 'danger');
                console.error('Error loading iterations:', error);
            });
    }

    /**
     * Enable the load button when an iteration is selected
     */
    function enableLoadButton() {
        loadPromptBtn.disabled = !iterationSelect.value;
    }

    /**
     * Load the prompts for the selected experiment and iteration
     */
    function loadSelectedPrompt() {
        const experimentId = experimentSelect.value;
        const iteration = iterationSelect.value;
        
        if (!experimentId || !iteration) {
            showAlert('Please select both an experiment and iteration.', 'warning');
            return;
        }
        
        showSpinner();
        
        fetch(`/experiments/${experimentId}/iterations/${iteration}/examples`)
            .then(response => response.json())
            .then(data => {
                hideSpinner();
                
                if (data.system_prompt && data.output_prompt) {
                    systemPromptTextarea.value = data.system_prompt;
                    outputPromptTextarea.value = data.output_prompt;
                    showAlert('Prompts loaded successfully.', 'success');
                } else {
                    showAlert('No prompts found for this iteration.', 'warning');
                }
            })
            .catch(error => {
                hideSpinner();
                showAlert(`Error loading prompts: ${error}`, 'danger');
                console.error('Error loading prompts:', error);
            });
    }

    /**
     * Handle dataset change event
     */
    function handleDatasetChange() {
        // You can add specific logic here if needed based on dataset selection
        // For example, adjusting batch size or metrics visibility
    }

    /**
     * Run the evaluation process
     */
    function runEvaluation() {
        // Get selected metrics
        const selectedMetrics = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);
            
        if (selectedMetrics.length === 0) {
            showAlert('Please select at least one metric for evaluation.', 'warning');
            return;
        }
        
        // Get form data
        const evaluationData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value,
            evaluation_system_prompt: evaluationSystemPromptTextarea.value,
            evaluation_output_prompt: evaluationOutputPromptTextarea.value,
            dataset_type: evalDatasetSelect.value,
            batch_size: parseInt(evalBatchSizeInput.value),
            metrics: selectedMetrics
        };
        
        if (!evaluationData.system_prompt || !evaluationData.output_prompt) {
            showAlert('System prompt and output prompt are required.', 'warning');
            return;
        }
        
        if (!evaluationData.evaluation_system_prompt || !evaluationData.evaluation_output_prompt) {
            showAlert('Evaluation system prompt and output prompt are required.', 'warning');
            return;
        }
        
        showSpinner();
        
        // This endpoint needs to be implemented in the backend
        fetch('/run_four_step_evaluation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(evaluationData)
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            hideSpinner();
            showAlert(`Error running evaluation: ${error}`, 'danger');
            console.error('Error running evaluation:', error);
        });
    }
    
    /**
     * Display evaluation results in the UI
     */
    function displayResults(data) {
        // Clear previous results
        resultsBody.innerHTML = '';
        
        if (!data.results || data.results.length === 0) {
            showAlert('No results to display.', 'warning');
            return;
        }
        
        // Update metrics
        const metrics = data.metrics || {};
        avgScoreDisplay.textContent = metrics.avg_score ? metrics.avg_score.toFixed(2) : '-';
        perfectMatchesDisplay.textContent = metrics.perfect_matches ? `${metrics.perfect_matches}/${metrics.total_examples}` : '-';
        totalExamplesDisplay.textContent = metrics.total_examples || '-';
        
        const matchRate = metrics.perfect_matches && metrics.total_examples 
            ? ((metrics.perfect_matches / metrics.total_examples) * 100).toFixed(1) + '%' 
            : '-';
        matchRateDisplay.textContent = matchRate;
        
        // Add results to table
        data.results.forEach((result, index) => {
            const row = document.createElement('tr');
            
            // Limit text length for display
            const userInput = limitText(result.user_input, 50);
            const groundTruth = limitText(result.ground_truth_output, 50);
            const modelResponse = limitText(result.model_response, 100);
            
            // Create cell for evaluation details
            let evaluationHtml = '-';
            if (result.evaluation_result) {
                try {
                    // Try to parse if it's a JSON string
                    const evalResult = typeof result.evaluation_result === 'string' 
                        ? JSON.parse(result.evaluation_result) 
                        : result.evaluation_result;
                        
                    evaluationHtml = `<strong>Score:</strong> ${evalResult.score}<br>`;
                    if (evalResult.reasoning) {
                        evaluationHtml += `<small>${limitText(evalResult.reasoning, 100)}</small>`;
                    }
                } catch (e) {
                    // If not valid JSON, just show as text
                    evaluationHtml = limitText(result.evaluation_result, 100);
                }
            }
            
            // Build the row
            row.innerHTML = `
                <td>${index + 1}</td>
                <td><div class="text-truncate">${userInput}</div></td>
                <td><div class="text-truncate">${groundTruth}</div></td>
                <td><div class="text-truncate">${modelResponse}</div></td>
                <td>${evaluationHtml}</td>
                <td><span class="badge ${getBadgeClass(result.score)}">${result.score.toFixed(2)}</span></td>
            `;
            
            resultsBody.appendChild(row);
        });
        
        // Update chart
        updateMetricsChart(data.metrics, data.results);
    }
    
    /**
     * Update the metrics chart with evaluation results
     */
    function updateMetricsChart(metrics, results) {
        // If chart exists, destroy it
        if (metricsChart) {
            metricsChart.destroy();
        }
        
        // Group scores into ranges
        const scoreRanges = {
            'Perfect (0.9-1.0)': 0,
            'Good (0.7-0.9)': 0,
            'Average (0.5-0.7)': 0,
            'Poor (0.3-0.5)': 0,
            'Bad (0-0.3)': 0
        };
        
        results.forEach(result => {
            const score = result.score;
            if (score >= 0.9) scoreRanges['Perfect (0.9-1.0)']++;
            else if (score >= 0.7) scoreRanges['Good (0.7-0.9)']++;
            else if (score >= 0.5) scoreRanges['Average (0.5-0.7)']++;
            else if (score >= 0.3) scoreRanges['Poor (0.3-0.5)']++;
            else scoreRanges['Bad (0-0.3)']++;
        });
        
        // Create the chart
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        metricsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(scoreRanges),
                datasets: [{
                    label: 'Number of Examples',
                    data: Object.values(scoreRanges),
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.7)',  // Perfect (teal)
                        'rgba(54, 162, 235, 0.7)',  // Good (blue)
                        'rgba(255, 206, 86, 0.7)',  // Average (yellow)
                        'rgba(255, 159, 64, 0.7)',  // Poor (orange)
                        'rgba(255, 99, 132, 0.7)'   // Bad (red)
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Save evaluation results
     */
    function saveEvaluationResults() {
        const evaluationData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value,
            evaluation_system_prompt: evaluationSystemPromptTextarea.value,
            evaluation_output_prompt: evaluationOutputPromptTextarea.value,
            metrics: getMetricsData()
        };
        
        showSpinner();
        
        // This endpoint needs to be implemented in the backend
        fetch('/save_evaluation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(evaluationData)
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }
            
            showAlert('Evaluation results saved successfully.', 'success');
        })
        .catch(error => {
            hideSpinner();
            showAlert(`Error saving evaluation results: ${error}`, 'danger');
            console.error('Error saving evaluation results:', error);
        });
    }
    
    /**
     * Optimize prompts based on evaluation results
     */
    function optimizePromptsFromEvaluation() {
        const optimizeData = {
            system_prompt: systemPromptTextarea.value,
            output_prompt: outputPromptTextarea.value,
            evaluation_system_prompt: evaluationSystemPromptTextarea.value,
            evaluation_output_prompt: evaluationOutputPromptTextarea.value,
            dataset_type: evalDatasetSelect.value,
            batch_size: parseInt(evalBatchSizeInput.value),
            metrics: getSelectedMetrics()
        };
        
        showSpinner();
        
        // This endpoint needs to be implemented in the backend
        fetch('/optimize_from_evaluation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(optimizeData)
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }
            
            // Redirect to final prompts page with the new prompts
            if (data.experiment_id) {
                window.location.href = `/final_prompts?experiment=${data.experiment_id}&iteration=${data.iteration}`;
            } else {
                showAlert('Optimization complete but no experiment ID returned.', 'warning');
            }
        })
        .catch(error => {
            hideSpinner();
            showAlert(`Error optimizing prompts: ${error}`, 'danger');
            console.error('Error optimizing prompts:', error);
        });
    }
    
    /**
     * Go to final prompts page with current prompts
     */
    function goToFinalPrompts() {
        const experimentId = experimentSelect.value;
        const iteration = iterationSelect.value;
        
        if (experimentId && iteration) {
            window.location.href = `/final_prompts?experiment=${experimentId}&iteration=${iteration}`;
        } else {
            // Save current prompts temporarily and redirect
            localStorage.setItem('temp_system_prompt', systemPromptTextarea.value);
            localStorage.setItem('temp_output_prompt', outputPromptTextarea.value);
            window.location.href = '/final_prompts';
        }
    }
    
    /**
     * Get the currently selected metrics
     */
    function getSelectedMetrics() {
        return Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);
    }
    
    /**
     * Get metrics data from the UI
     */
    function getMetricsData() {
        return {
            avg_score: parseFloat(avgScoreDisplay.textContent) || 0,
            perfect_matches: parseInt(perfectMatchesDisplay.textContent.split('/')[0]) || 0,
            total_examples: parseInt(totalExamplesDisplay.textContent) || 0
        };
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
    
    /**
     * Limit text length with ellipsis
     */
    function limitText(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }
    
    /**
     * Get the appropriate badge class based on score
     */
    function getBadgeClass(score) {
        if (score >= 0.9) return 'bg-success';
        if (score >= 0.7) return 'bg-primary';
        if (score >= 0.5) return 'bg-info';
        if (score >= 0.3) return 'bg-warning';
        return 'bg-danger';
    }
});