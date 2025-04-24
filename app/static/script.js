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
    
    // Dashboard elements
    const trainCountElement = document.getElementById('train-count');
    const validationCountElement = document.getElementById('validation-count');
    const latestExperimentElement = document.getElementById('latest-experiment');
    const iterationCountElement = document.getElementById('iteration-count');
    const bestScoreElement = document.getElementById('best-score');
    const quickTestButton = document.getElementById('quick-test-btn');
    const loadNejmButton = document.getElementById('load-nejm-btn');
    const clearExamplesButton = document.getElementById('clear-examples-btn');
    const metricsChart = document.getElementById('metrics-chart');
    
    // Initialize dashboard components
    initDashboard();
    initMetricsChart();
    
    // Register additional event listeners for dashboard components
    if (quickTestButton) quickTestButton.addEventListener('click', performQuickTest);
    if (loadNejmButton) loadNejmButton.addEventListener('click', loadNejmData);
    if (clearExamplesButton) clearExamplesButton.addEventListener('click', clearExamplesData);
    
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
    
    // Dashboard Functions
    
    /**
     * Initialize dashboard components with data
     */
    function initDashboard() {
        // Set a flag to prevent multiple initializations on the same page
        if (window.dashboardInitialized) {
            return;
        }
        window.dashboardInitialized = true;
        
        // Fetch dataset counts
        fetchDatasetCounts();
        
        // Fetch latest experiment data
        fetchExperimentData();
        
        // Initialize accuracy metrics display
        const trainingAccuracy = document.getElementById('training-accuracy');
        const validationAccuracy = document.getElementById('validation-accuracy');
        const optimizerImprovement = document.getElementById('optimizer-improvement');
        
        if (trainingAccuracy && validationAccuracy && optimizerImprovement) {
            // Initially show placeholder values - these would be updated with real data from API
            trainingAccuracy.textContent = '0.88';
            validationAccuracy.textContent = '0.84';
            optimizerImprovement.textContent = '+12%';
            
            // Fetch real metrics data only once
            fetch('/api/metrics_summary')
                .then(response => response.json())
                .then(data => {
                    if (data.error) return;
                    
                    if (data.training_accuracy) {
                        trainingAccuracy.textContent = data.training_accuracy.toFixed(2);
                    }
                    if (data.validation_accuracy) {
                        validationAccuracy.textContent = data.validation_accuracy.toFixed(2);
                    }
                    if (data.improvement_percentage) {
                        optimizerImprovement.textContent = `+${data.improvement_percentage}%`;
                    }
                })
                .catch(error => {
                    console.error('Error fetching metrics summary:', error);
                });
        }
    }
    
    /**
     * Initialize the metrics chart on the dashboard
     */
    function initMetricsChart() {
        if (!metricsChart) return;
        
        // Destroy existing chart instance if it exists
        if (window.dashboardMetricsChart && typeof window.dashboardMetricsChart.destroy === 'function') {
            window.dashboardMetricsChart.destroy();
        }
        
        // Clear the canvas manually for safety
        const parent = metricsChart.parentNode;
        if (parent) {
            const newCanvas = document.createElement('canvas');
            newCanvas.id = 'metrics-chart';
            newCanvas.className = metricsChart.className;
            newCanvas.height = metricsChart.height;
            parent.replaceChild(newCanvas, metricsChart);
            metricsChart = newCanvas;
        }
        
        const ctx = metricsChart.getContext('2d');
        window.dashboardMetricsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Original', 'Evaluator', 'Optimizer', 'Validation', 'Final'],
                datasets: [
                    {
                        label: 'Training Accuracy',
                        data: [0.75, 0.82, 0.88, 0.90, 0.92],
                        backgroundColor: 'rgba(67, 97, 238, 0.6)',
                        borderColor: 'rgba(67, 97, 238, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Validation Accuracy',
                        data: [0.72, 0.79, 0.82, 0.86, 0.89],
                        backgroundColor: 'rgba(76, 201, 240, 0.6)',
                        borderColor: 'rgba(76, 201, 240, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            boxWidth: 8
                        }
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                const phaseMap = {
                                    'Original': 'Step 1: Data Preparation (Vertex AI)',
                                    'Evaluator': 'Step 2: Evaluation (Hugging Face)',
                                    'Optimizer': 'Step 3: Optimization (Vertex AI)',
                                    'Validation': 'Step 4: Validation (Hugging Face)',
                                    'Final': 'Step 5: Finalization (Vertex AI)'
                                };
                                return phaseMap[tooltipItems[0].label] || tooltipItems[0].label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.7,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Accuracy'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '5-API Workflow Steps'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Fetch dataset counts for the dashboard
     */
    function fetchDatasetCounts() {
        if (!trainCountElement || !validationCountElement) return;
        
        // Fetch training count
        fetch('/load_dataset?type=train')
            .then(response => response.json())
            .then(data => {
                if (data.count) {
                    trainCountElement.textContent = data.count;
                }
            })
            .catch(error => {
                console.error('Error fetching training counts:', error);
            });
            
        // Fetch validation count
        fetch('/load_dataset?type=validation')
            .then(response => response.json())
            .then(data => {
                if (data.count) {
                    validationCountElement.textContent = data.count;
                }
            })
            .catch(error => {
                console.error('Error fetching validation counts:', error);
            });
    }
    
    /**
     * Fetch experiment data for the dashboard
     */
    function fetchExperimentData() {
        if (!latestExperimentElement || !iterationCountElement || !bestScoreElement) return;
        
        fetch('/experiments')
            .then(response => response.json())
            .then(data => {
                if (data.experiments && data.experiments.length > 0) {
                    // Sort by date descending
                    const sortedExperiments = [...data.experiments].sort((a, b) => 
                        new Date(b.created_at) - new Date(a.created_at)
                    );
                    
                    const latestExperiment = sortedExperiments[0];
                    latestExperimentElement.textContent = latestExperiment.id;
                    
                    // Fetch experiment details
                    fetch(`/experiments/${latestExperiment.id}`)
                        .then(response => response.json())
                        .then(expData => {
                            if (expData.iterations && expData.iterations.length > 0) {
                                iterationCountElement.textContent = expData.iterations.length;
                                
                                // Find best score
                                const bestScore = Math.max(...expData.iterations.map(i => i.avg_score));
                                bestScoreElement.textContent = bestScore.toFixed(2);
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching experiment details:', error);
                        });
                }
            })
            .catch(error => {
                console.error('Error fetching experiments:', error);
            });
    }
    
    /**
     * Perform a quick test on a sample
     */
    function performQuickTest() {
        // Use a sample prompt for testing
        const sampleInput = "What is the primary treatment for acute myocardial infarction?";
        
        const systemPrompt = systemPromptTextarea.value.trim();
        const outputPrompt = outputPromptTextarea.value.trim();
        
        if (!systemPrompt || !outputPrompt) {
            showAlert('Please enter system and output prompts first');
            return;
        }
        
        // Show spinner
        spinnerElement.style.display = 'block';
        
        // Prepare test data
        const testData = {
            system_prompt: systemPrompt,
            output_prompt: outputPrompt,
            user_input: sampleInput
        };
        
        // Call the test API
        fetch('/test_prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(testData)
        })
        .then(response => response.json())
        .then(data => {
            spinnerElement.style.display = 'none';
            
            if (data.error) {
                showAlert(data.error);
                return;
            }
            
            // Show the result in an alert
            const result = `
                <strong>Quick Test Result:</strong><br>
                <div class="mt-2 p-2 bg-light rounded">
                    <strong>Input:</strong> ${sampleInput}<br><br>
                    <strong>Response:</strong><br>
                    ${data.response}
                </div>
                <div class="mt-2">
                    <span class="badge bg-info">${data.response_time}</span>
                    <span class="badge bg-primary">${data.quality}</span>
                </div>
            `;
            
            showAlert(result, 'success');
        })
        .catch(error => {
            spinnerElement.style.display = 'none';
            showAlert('Error running test: ' + error.message);
        });
    }
    
    /**
     * Load NEJM dataset
     */
    function loadNejmData() {
        // Show spinner
        spinnerElement.style.display = 'block';
        
        // First load NEJM prompts
        fetch('/load_dataset?type=nejm_prompts')
            .then(response => response.json())
            .then(data => {
                if (data.prompts) {
                    systemPromptTextarea.value = data.prompts.system_prompt;
                    outputPromptTextarea.value = data.prompts.output_prompt;
                }
                
                // Then load NEJM examples
                return fetch('/load_dataset?type=nejm_train');
            })
            .then(response => response.json())
            .then(data => {
                spinnerElement.style.display = 'none';
                
                if (data.error) {
                    showAlert(data.error);
                    return;
                }
                
                if (data.examples && data.examples.length > 0) {
                    // Use the truncated CSV content for display
                    examplesTextarea.value = data.csv_content;
                    showAlert(`Successfully loaded ${data.count} NEJM examples`, 'success');
                } else {
                    showAlert('No NEJM examples found');
                }
            })
            .catch(error => {
                spinnerElement.style.display = 'none';
                showAlert('Error loading NEJM data: ' + error.message);
            });
    }
    
    /**
     * Clear examples textarea
     */
    function clearExamplesData() {
        examplesTextarea.value = '';
        showAlert('Examples cleared', 'info');
    }
});
