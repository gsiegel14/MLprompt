document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const experimentsBodyEl = document.getElementById('experiments-body');
    const experimentDetailsEl = document.getElementById('experiment-details');
    const compareViewEl = document.getElementById('compare-view');
    const backToListBtn = document.getElementById('back-to-list');
    const backToDetailsBtn = document.getElementById('back-to-details');
    const currentExperimentIdEl = document.getElementById('current-experiment-id');
    const iterationsAccordionEl = document.getElementById('iterations-accordion');
    const spinner = document.getElementById('spinner');
    
    // Setup Chart.js
    let historyChart;
    setupHistoryChart();
    
    // Initial state variables
    let currentExperimentId = null;
    let currentExperimentData = null;
    
    // Load all experiments
    loadExperiments();
    
    // Event listeners
    backToListBtn.addEventListener('click', showExperimentsList);
    backToDetailsBtn.addEventListener('click', showExperimentDetails);
    
    // Functions
    function loadExperiments() {
        showSpinner();
        fetch('/experiments')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                } else if (data.experiments && data.experiments.length > 0) {
                    populateExperimentsTable(data.experiments);
                } else {
                    experimentsBodyEl.innerHTML = `
                        <tr>
                            <td colspan="6" class="text-center py-4">
                                <div class="text-muted">
                                    <i class="fa-solid fa-folder-open me-2 fs-4"></i>
                                    <p class="mb-1">No experiments found</p>
                                    <small>Start a new experiment in the Training interface</small>
                                </div>
                            </td>
                        </tr>
                    `;
                }
            })
            .catch(error => {
                console.error('Error loading experiments:', error);
                showAlert('Error loading experiments', 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    function populateExperimentsTable(experiments) {
        experimentsBodyEl.innerHTML = '';
        
        experiments.forEach(exp => {
            const date = new Date(exp.timestamp * 1000);
            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            
            const avgScore = exp.metrics?.avg_score || 0;
            const perfectMatchPercent = exp.metrics?.perfect_match_percent || 0;
            
            // Calculate improvement (would need more data in a real implementation)
            const improvement = '+0.0%';
            const improvementClass = 'text-success';
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${exp.experiment_id}</td>
                <td>${formattedDate}</td>
                <td>${exp.iteration + 1}</td>
                <td>${(avgScore * 100).toFixed(1)}%</td>
                <td class="${improvementClass}">${improvement}</td>
                <td>
                    <button class="btn btn-sm btn-primary view-experiment" data-id="${exp.experiment_id}">
                        <i class="fa-solid fa-eye me-1"></i> View
                    </button>
                </td>
            `;
            experimentsBodyEl.appendChild(row);
        });
        
        // Add event listeners
        document.querySelectorAll('.view-experiment').forEach(btn => {
            btn.addEventListener('click', function() {
                const expId = this.getAttribute('data-id');
                loadExperimentDetails(expId);
            });
        });
    }
    
    function loadExperimentDetails(experimentId) {
        showSpinner();
        fetch(`/experiments/${experimentId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                } else if (data.iterations && data.iterations.length > 0) {
                    currentExperimentId = experimentId;
                    currentExperimentData = data;
                    
                    // Update UI
                    currentExperimentIdEl.textContent = experimentId;
                    
                    // Populate iterations
                    populateIterationsAccordion(data.iterations);
                    
                    // Update chart
                    updateHistoryChart(data.iterations);
                    
                    // Show details view
                    showExperimentDetails();
                } else {
                    showAlert(`No iterations found for experiment ${experimentId}`, 'warning');
                }
            })
            .catch(error => {
                console.error('Error loading experiment details:', error);
                showAlert('Error loading experiment details', 'danger');
            })
            .finally(() => {
                hideSpinner();
            });
    }
    
    function populateIterationsAccordion(iterations) {
        iterationsAccordionEl.innerHTML = '';
        
        iterations.forEach((iteration, index) => {
            const avgScore = iteration.metrics?.avg_score || 0;
            const perfectMatches = iteration.metrics?.perfect_matches || 0;
            const totalExamples = iteration.metrics?.total_examples || 0;
            
            // Calculate improvements
            const previousScore = index > 0 ? iterations[index - 1].metrics?.avg_score || 0 : 0;
            const scoreImprovement = avgScore - previousScore;
            const improvementClass = scoreImprovement > 0 ? 'text-success' : (scoreImprovement < 0 ? 'text-danger' : 'text-muted');
            const improvementSign = scoreImprovement > 0 ? '+' : '';
            
            const item = document.createElement('div');
            item.className = 'accordion-item';
            item.innerHTML = `
                <h2 class="accordion-header">
                    <button class="accordion-button ${index === iterations.length - 1 ? '' : 'collapsed'}" type="button" 
                            data-bs-toggle="collapse" data-bs-target="#iteration-${index}">
                        <div class="d-flex justify-content-between align-items-center w-100 me-3">
                            <span><strong>Iteration ${iteration.iteration}</strong></span>
                            <span class="badge bg-primary ms-2">${(avgScore * 100).toFixed(1)}% Score</span>
                            <span class="${improvementClass}">${improvementSign}${(scoreImprovement * 100).toFixed(1)}%</span>
                        </div>
                    </button>
                </h2>
                <div id="iteration-${index}" class="accordion-collapse collapse ${index === iterations.length - 1 ? 'show' : ''}">
                    <div class="accordion-body">
                        <div class="row mb-3">
                            <div class="col-md-4 text-center">
                                <div class="fs-4">${(avgScore * 100).toFixed(1)}%</div>
                                <div class="text-muted">Average Score</div>
                            </div>
                            <div class="col-md-4 text-center">
                                <div class="fs-4">${perfectMatches}/${totalExamples}</div>
                                <div class="text-muted">Perfect Matches</div>
                            </div>
                            <div class="col-md-4 text-center">
                                <div class="fs-4">${(iteration.metrics?.perfect_match_percent || 0).toFixed(1)}%</div>
                                <div class="text-muted">Perfect Match %</div>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <h6>System Prompt:</h6>
                            <pre class="p-2 bg-dark rounded">${escapeHtml(iteration.system_prompt).substring(0, 200)}${iteration.system_prompt.length > 200 ? '...' : ''}</pre>
                        </div>
                        
                        <div class="mb-3">
                            <h6>Output Prompt:</h6>
                            <pre class="p-2 bg-dark rounded">${escapeHtml(iteration.output_prompt).substring(0, 200)}${iteration.output_prompt.length > 200 ? '...' : ''}</pre>
                        </div>
                        
                        ${index > 0 ? `
                        <div class="mb-3">
                            <h6>Optimizer Reasoning:</h6>
                            <div class="p-2 bg-dark rounded">
                                <small>${iteration.reasoning ? escapeHtml(iteration.reasoning).substring(0, 300) + (iteration.reasoning.length > 300 ? '...' : '') : 'No reasoning available'}</small>
                            </div>
                        </div>
                        ` : ''}
                        
                        <div class="mt-3 text-center">
                            ${index > 0 ? `
                            <button class="btn btn-sm btn-outline-info compare-prompts" data-current="${index}" data-previous="${index - 1}">
                                <i class="fa-solid fa-code-compare me-1"></i> Compare with Previous
                            </button>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
            
            iterationsAccordionEl.appendChild(item);
        });
        
        // Add event listeners to compare buttons
        document.querySelectorAll('.compare-prompts').forEach(btn => {
            btn.addEventListener('click', function() {
                const currentIndex = parseInt(this.getAttribute('data-current'));
                const previousIndex = parseInt(this.getAttribute('data-previous'));
                
                showPromptComparison(previousIndex, currentIndex);
            });
        });
    }
    
    function showPromptComparison(originalIndex, optimizedIndex) {
        const iterations = currentExperimentData.iterations;
        if (!iterations || originalIndex < 0 || optimizedIndex >= iterations.length) {
            showAlert('Invalid comparison', 'danger');
            return;
        }
        
        const original = iterations[originalIndex];
        const optimized = iterations[optimizedIndex];
        
        // Update UI elements
        document.getElementById('original-iteration').textContent = `Iteration ${original.iteration}`;
        document.getElementById('optimized-iteration').textContent = `Iteration ${optimized.iteration}`;
        
        document.getElementById('original-system-prompt').textContent = original.system_prompt;
        document.getElementById('optimized-system-prompt').textContent = optimized.system_prompt;
        
        document.getElementById('original-output-prompt').textContent = original.output_prompt;
        document.getElementById('optimized-output-prompt').textContent = optimized.output_prompt;
        
        document.getElementById('original-score').textContent = `${(original.metrics?.avg_score * 100 || 0).toFixed(1)}%`;
        document.getElementById('optimized-score').textContent = `${(optimized.metrics?.avg_score * 100 || 0).toFixed(1)}%`;
        
        document.getElementById('optimizer-reasoning').textContent = optimized.reasoning || 'No reasoning available';
        
        // Show comparison view
        experimentDetailsEl.style.display = 'none';
        compareViewEl.style.display = 'block';
    }
    
    function setupHistoryChart() {
        const ctx = document.getElementById('history-chart').getContext('2d');
        historyChart = new Chart(ctx, {
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
    
    function updateHistoryChart(iterations) {
        const labels = iterations.map(item => `Iteration ${item.iteration}`);
        const avgScores = iterations.map(item => (item.metrics?.avg_score || 0) * 100);
        const perfectMatches = iterations.map(item => item.metrics?.perfect_match_percent || 0);
        
        historyChart.data.labels = labels;
        historyChart.data.datasets[0].data = avgScores;
        historyChart.data.datasets[1].data = perfectMatches;
        historyChart.update();
    }
    
    function showExperimentsList() {
        experimentDetailsEl.style.display = 'none';
        compareViewEl.style.display = 'none';
    }
    
    function showExperimentDetails() {
        experimentDetailsEl.style.display = 'block';
        compareViewEl.style.display = 'none';
    }
    
    function showSpinner() {
        spinner.style.display = 'flex';
    }
    
    function hideSpinner() {
        spinner.style.display = 'none';
    }
    
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
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});