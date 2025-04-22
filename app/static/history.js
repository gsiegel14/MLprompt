document.addEventListener("DOMContentLoaded", function() {
    // If experiments and iterations are loaded from server-side, display them
    if (typeof initialData !== 'undefined' && initialData.experiments) {
        displayExperiments(initialData.experiments);
    } else {
        fetchExperiments();
    }
});

function fetchExperiments() {
    fetch('/api/experiments')
        .then(response => response.json())
        .then(data => {
            displayExperiments(data.experiments);
        })
        .catch(error => {
            console.error('Error fetching experiments:', error);
            document.getElementById('loading-spinner').style.display = 'none';
            document.getElementById('error-message').textContent = 'Failed to load experiments. Please try again later.';
            document.getElementById('error-message').style.display = 'block';
        });
}

function displayExperiments(experiments) {
    const experimentsTable = document.getElementById('experiments-table');
    const loadingSpinner = document.getElementById('loading-spinner');

    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
    }

    if (!experiments || experiments.length === 0) {
        document.getElementById('no-experiments').style.display = 'block';
        return;
    }

    const tbody = experimentsTable.querySelector('tbody');
    tbody.innerHTML = '';

    experiments.forEach((experiment, index) => {
        const row = document.createElement('tr');

        const dateCell = document.createElement('td');
        dateCell.textContent = formatDate(experiment.id);

        const iterationsCell = document.createElement('td');
        iterationsCell.textContent = experiment.iterations ? experiment.iterations.length : 0;

        const scoreCell = document.createElement('td');
        if (experiment.best_score !== undefined) {
            scoreCell.textContent = (experiment.best_score * 100).toFixed(1) + '%';
        } else {
            scoreCell.textContent = 'N/A';
        }

        const actionsCell = document.createElement('td');
        const viewButton = document.createElement('button');
        viewButton.textContent = 'View';
        viewButton.className = 'btn btn-sm btn-primary view-experiment';
        viewButton.setAttribute('data-id', experiment.id);
        viewButton.onclick = function() {
            showExperimentDetails(experiment.id);
        };
        actionsCell.appendChild(viewButton);

        row.appendChild(dateCell);
        row.appendChild(iterationsCell);
        row.appendChild(scoreCell);
        row.appendChild(actionsCell);

        tbody.appendChild(row);
    });

    experimentsTable.style.display = 'table';
}

function formatDate(experimentId) {
    // Assuming experimentId format is YYYYMMDD_HHMMSS
    if (!experimentId || typeof experimentId !== 'string' || experimentId.length < 15) {
        return experimentId;
    }

    try {
        const year = experimentId.substring(0, 4);
        const month = experimentId.substring(4, 6);
        const day = experimentId.substring(6, 8);
        const hour = experimentId.substring(9, 11);
        const minute = experimentId.substring(11, 13);

        return `${year}-${month}-${day} ${hour}:${minute}`;
    } catch (e) {
        return experimentId;
    }
}

function showExperimentDetails(experimentId) {
    // Clear previous content and show loading indicator
    document.getElementById('experiment-details').style.display = 'block';
    document.getElementById('experiment-details-loading').style.display = 'block';
    document.getElementById('iterations-container').style.display = 'none';
    document.getElementById('breadcrumb-experiment').textContent = experimentId;

    // Scroll to experiment details
    document.getElementById('experiment-details').scrollIntoView({
        behavior: 'smooth'
    });

    fetch(`/api/experiments/${experimentId}`)
        .then(response => response.json())
        .then(data => {
            displayIterations(experimentId, data.iterations);
        })
        .catch(error => {
            console.error('Error fetching experiment details:', error);
            document.getElementById('experiment-details-loading').style.display = 'none';
            document.getElementById('experiment-details-error').textContent = 'Failed to load experiment details. Please try again later.';
            document.getElementById('experiment-details-error').style.display = 'block';
        });
}

function displayIterations(experimentId, iterations) {
    const container = document.getElementById('iterations-accordion');
    container.innerHTML = '';

    if (!iterations || iterations.length === 0) {
        document.getElementById('experiment-details-loading').style.display = 'none';
        document.getElementById('no-iterations').style.display = 'block';
        return;
    }

    iterations.forEach((iteration, index) => {
        const card = document.createElement('div');
        card.className = 'card mb-3';

        const cardHeader = document.createElement('div');
        cardHeader.className = 'card-header d-flex justify-content-between align-items-center';
        cardHeader.id = `heading-${iteration.iteration}`;

        const headerButton = document.createElement('button');
        headerButton.className = 'btn btn-link';
        headerButton.setAttribute('data-toggle', 'collapse');
        headerButton.setAttribute('data-target', `#collapse-${iteration.iteration}`);
        headerButton.setAttribute('aria-expanded', 'false');
        headerButton.setAttribute('aria-controls', `collapse-${iteration.iteration}`);

        let headerText = `Iteration ${iteration.iteration}`;
        if (iteration.metrics && iteration.metrics.avg_score !== undefined) {
            headerText += ` - Score: ${(iteration.metrics.avg_score * 100).toFixed(1)}%`;
        }
        headerButton.textContent = headerText;

        const headerActions = document.createElement('div');

        const viewExamplesBtn = document.createElement('button');
        viewExamplesBtn.className = 'btn btn-sm btn-outline-primary view-examples';
        viewExamplesBtn.setAttribute('data-iteration', iteration.iteration);
        viewExamplesBtn.setAttribute('onclick', `window.loadExamplesForIteration(${iteration.iteration})`);
        viewExamplesBtn.textContent = 'View Examples';

        headerActions.appendChild(viewExamplesBtn);
        cardHeader.appendChild(headerButton);
        cardHeader.appendChild(headerActions);

        const cardCollapse = document.createElement('div');
        cardCollapse.id = `collapse-${iteration.iteration}`;
        cardCollapse.className = 'collapse';
        cardCollapse.setAttribute('aria-labelledby', `heading-${iteration.iteration}`);
        cardCollapse.setAttribute('data-parent', '#iterations-accordion');

        const cardBody = document.createElement('div');
        cardBody.className = 'card-body';

        // System Prompt
        const systemPromptSection = document.createElement('div');
        systemPromptSection.className = 'mb-4';

        const systemPromptHeader = document.createElement('h5');
        systemPromptHeader.textContent = 'System Prompt';

        const systemPromptContent = document.createElement('pre');
        systemPromptContent.className = 'prompt-content';
        systemPromptContent.textContent = iteration.system_prompt || 'No system prompt available';

        systemPromptSection.appendChild(systemPromptHeader);
        systemPromptSection.appendChild(systemPromptContent);

        // Output Prompt
        const outputPromptSection = document.createElement('div');
        outputPromptSection.className = 'mb-4';

        const outputPromptHeader = document.createElement('h5');
        outputPromptHeader.textContent = 'Output Prompt';

        const outputPromptContent = document.createElement('pre');
        outputPromptContent.className = 'prompt-content';
        outputPromptContent.textContent = iteration.output_prompt || 'No output prompt available';

        outputPromptSection.appendChild(outputPromptHeader);
        outputPromptSection.appendChild(outputPromptContent);

        // Metrics
        const metricsSection = document.createElement('div');
        metricsSection.className = 'mb-4';

        const metricsHeader = document.createElement('h5');
        metricsHeader.textContent = 'Metrics';

        const metricsContent = document.createElement('div');
        metricsContent.className = 'metrics-content';

        if (iteration.metrics) {
            const metrics = iteration.metrics;

            const metricsList = document.createElement('ul');
            metricsList.className = 'list-group';

            for (const [key, value] of Object.entries(metrics)) {
                const metricItem = document.createElement('li');
                metricItem.className = 'list-group-item d-flex justify-content-between align-items-center';

                const metricKey = document.createElement('span');
                metricKey.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                const metricValue = document.createElement('span');
                if (typeof value === 'number') {
                    if (key.includes('percent') || key.includes('score')) {
                        metricValue.textContent = `${(value * 100).toFixed(1)}%`;
                    } else {
                        metricValue.textContent = value;
                    }
                } else {
                    metricValue.textContent = value;
                }

                metricItem.appendChild(metricKey);
                metricItem.appendChild(metricValue);
                metricsList.appendChild(metricItem);
            }

            metricsContent.appendChild(metricsList);
        } else {
            metricsContent.textContent = 'No metrics available';
        }

        metricsSection.appendChild(metricsHeader);
        metricsSection.appendChild(metricsContent);

        // Optimizer Reasoning
        if (iteration.optimizer_reasoning) {
            const reasoningSection = document.createElement('div');
            reasoningSection.className = 'mb-4';

            const reasoningHeader = document.createElement('h5');
            reasoningHeader.textContent = 'Optimizer Reasoning';

            const reasoningContent = document.createElement('pre');
            reasoningContent.className = 'prompt-content';
            reasoningContent.textContent = iteration.optimizer_reasoning;

            reasoningSection.appendChild(reasoningHeader);
            reasoningSection.appendChild(reasoningContent);

            cardBody.appendChild(reasoningSection);
        }

        cardBody.appendChild(systemPromptSection);
        cardBody.appendChild(outputPromptSection);
        cardBody.appendChild(metricsSection);

        cardCollapse.appendChild(cardBody);

        card.appendChild(cardHeader);
        card.appendChild(cardCollapse);

        container.appendChild(card);
    });

    document.getElementById('experiment-details-loading').style.display = 'none';
    document.getElementById('iterations-container').style.display = 'block';

    // Add examples container if it doesn't exist
    if (!document.getElementById('examples-container')) {
        const examplesContainer = document.createElement('div');
        examplesContainer.id = 'examples-container';
        examplesContainer.className = 'mt-4';
        examplesContainer.style.display = 'none';

        const examplesTitle = document.createElement('h4');
        examplesTitle.textContent = 'Examples';

        const examplesLoading = document.createElement('div');
        examplesLoading.id = 'examples-loading';
        examplesLoading.className = 'text-center my-4';
        examplesLoading.style.display = 'none';

        const spinner = document.createElement('div');
        spinner.className = 'spinner-border';
        spinner.setAttribute('role', 'status');

        const spinnerText = document.createElement('span');
        spinnerText.className = 'sr-only';
        spinnerText.textContent = 'Loading...';

        spinner.appendChild(spinnerText);
        examplesLoading.appendChild(spinner);

        const examplesContent = document.createElement('div');
        examplesContent.id = 'examples-content';

        examplesContainer.appendChild(examplesTitle);
        examplesContainer.appendChild(examplesLoading);
        examplesContainer.appendChild(examplesContent);

        document.getElementById('experiment-details').appendChild(examplesContainer);
    }

    // Update breadcrumb
    document.getElementById('breadcrumb-experiment').textContent = experimentId;
}

// Make sure this function is available in the global scope (window)
window.loadExamplesForIteration = function(iteration) {
    const experimentId = document.getElementById('breadcrumb-experiment')?.textContent;
    if (!experimentId) {
        console.error('Error: breadcrumb-experiment element not found');
        return;
    }

    // Show examples container and loading indicator
    const examplesContainer = document.getElementById('examples-container');
    const examplesLoading = document.getElementById('examples-loading');
    const examplesContent = document.getElementById('examples-content');
    
    if (!examplesContainer || !examplesLoading || !examplesContent) {
        console.error('Error: examples container elements not found');
        return;
    }

    examplesContainer.style.display = 'block';
    examplesLoading.style.display = 'block';
    examplesContent.innerHTML = '';

    // Update breadcrumb
    const breadcrumbIteration = document.getElementById('breadcrumb-iteration');
    if (breadcrumbIteration) {
        breadcrumbIteration.textContent = iteration;
    }

    // Scroll to examples container
    examplesContainer.scrollIntoView({
        behavior: 'smooth'
    });

    console.log(`Fetching examples for experiment ${experimentId}, iteration ${iteration}`);
    fetch(`/api/experiments/${experimentId}/iterations/${iteration}/examples`)
        .then(response => response.json())
        .then(data => {
            if (typeof displayExamples === 'function') {
                displayExamples(data.examples);
            } else {
                console.error('Error: displayExamples function not found');
                // Fallback display logic
                if (data.examples && data.examples.length > 0) {
                    examplesLoading.style.display = 'none';
                    const examplesHtml = data.examples.map((example, index) => {
                        return `
                            <div class="col-md-6 mb-4">
                                <div class="card example-card">
                                    <div class="card-body">
                                        <h5 class="card-title">Example ${index + 1}</h5>
                                        <div class="mb-3">
                                            <h6>User Input:</h6>
                                            <div class="user-input p-2 bg-light">${example.user_input || 'No user input'}</div>
                                        </div>
                                        <div class="mb-3">
                                            <h6>Ground Truth:</h6>
                                            <div class="ground-truth p-2 bg-light">${example.ground_truth_output || 'No ground truth'}</div>
                                        </div>
                                        <div class="mb-3">
                                            <h6>Model Response:</h6>
                                            <div class="model-response p-2 bg-light">${example.model_response || 'No model response'}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('');
                    
                    examplesContent.innerHTML = `<div class="row">${examplesHtml}</div>`;
                } else {
                    examplesLoading.style.display = 'none';
                    examplesContent.innerHTML = '<div class="alert alert-info">No examples available for this iteration.</div>';
                }
            }
        })
        .catch(error => {
            console.error('Error fetching examples:', error);
            examplesLoading.style.display = 'none';
            examplesContent.innerHTML = '<div class="alert alert-danger">Failed to load examples. Please try again later.</div>';
        });
};

function displayExamples(examples) {
    const examplesLoading = document.getElementById('examples-loading');
    const examplesContent = document.getElementById('examples-content');

    examplesLoading.style.display = 'none';

    if (!examples || examples.length === 0) {
        examplesContent.innerHTML = '<div class="alert alert-info">No examples available for this iteration.</div>';
        return;
    }

    const examplesGrid = document.createElement('div');
    examplesGrid.className = 'row';

    examples.forEach((example, index) => {
        const exampleCol = document.createElement('div');
        exampleCol.className = 'col-md-6 mb-4';

        const exampleCard = document.createElement('div');
        exampleCard.className = 'card example-card';

        const cardBody = document.createElement('div');
        cardBody.className = 'card-body';

        // Example number and score
        const cardHeader = document.createElement('div');
        cardHeader.className = 'd-flex justify-content-between align-items-center mb-3';

        const exampleNumber = document.createElement('h5');
        exampleNumber.className = 'card-title mb-0';
        exampleNumber.textContent = `Example ${index + 1}`;

        const scoreContainer = document.createElement('div');
        scoreContainer.className = 'score-container';

        const scoreValue = document.createElement('span');
        scoreValue.className = 'score badge ' + (example.score >= 0.9 ? 'badge-success' : (example.score >= 0.7 ? 'badge-warning' : 'badge-danger'));

        if (example.score !== undefined) {
            scoreValue.textContent = `Score: ${(example.score * 100).toFixed(1)}%`;
        } else {
            scoreValue.textContent = 'No Score';
            scoreValue.className = 'score badge badge-secondary';
        }

        scoreContainer.appendChild(scoreValue);
        cardHeader.appendChild(exampleNumber);
        cardHeader.appendChild(scoreContainer);

        // User Input
        const userInputSection = document.createElement('div');
        userInputSection.className = 'mb-3';

        const userInputLabel = document.createElement('h6');
        userInputLabel.textContent = 'User Input';

        const userInputContent = document.createElement('div');
        userInputContent.className = 'user-input p-2 bg-light border rounded';
        userInputContent.textContent = example.user_input || 'No user input available';

        userInputSection.appendChild(userInputLabel);
        userInputSection.appendChild(userInputContent);

        // Ground Truth
        const groundTruthSection = document.createElement('div');
        groundTruthSection.className = 'mb-3';

        const groundTruthLabel = document.createElement('h6');
        groundTruthLabel.textContent = 'Ground Truth';

        const groundTruthContent = document.createElement('div');
        groundTruthContent.className = 'ground-truth p-2 bg-light border rounded';
        groundTruthContent.textContent = example.ground_truth_output || 'No ground truth available';

        groundTruthSection.appendChild(groundTruthLabel);
        groundTruthSection.appendChild(groundTruthContent);

        // Model Response
        const modelResponseSection = document.createElement('div');
        modelResponseSection.className = 'mb-3';

        const modelResponseLabel = document.createElement('h6');
        modelResponseLabel.textContent = 'Model Response';

        const modelResponseContent = document.createElement('div');
        modelResponseContent.className = 'model-response p-2 bg-light border rounded';
        modelResponseContent.textContent = example.model_response || 'No model response available';

        modelResponseSection.appendChild(modelResponseLabel);
        modelResponseSection.appendChild(modelResponseContent);

        // Optimized Response (if available)
        if (example.optimized_response) {
            const optimizedResponseSection = document.createElement('div');
            optimizedResponseSection.className = 'mb-3';

            const optimizedResponseLabel = document.createElement('h6');
            optimizedResponseLabel.textContent = 'Optimized Response';

            const optimizedResponseContent = document.createElement('div');
            optimizedResponseContent.className = 'optimized-response p-2 bg-light border rounded';
            optimizedResponseContent.textContent = example.optimized_response;

            optimizedResponseSection.appendChild(optimizedResponseLabel);
            optimizedResponseSection.appendChild(optimizedResponseContent);

            cardBody.appendChild(optimizedResponseSection);
        }

        cardBody.appendChild(cardHeader);
        cardBody.appendChild(userInputSection);
        cardBody.appendChild(groundTruthSection);
        cardBody.appendChild(modelResponseSection);

        exampleCard.appendChild(cardBody);
        exampleCol.appendChild(exampleCard);
        examplesGrid.appendChild(exampleCol);
    });

    examplesContent.appendChild(examplesGrid);
}

//This part is unchanged from original
    // Setup Chart.js
    let historyChart;
    setupHistoryChart();

    // Initial state variables
    let currentExperimentId = null;
    let currentExperimentData = null;

    // Event listeners
    backToListBtn.addEventListener('click', showExperimentsList);
    backToDetailsBtn.addEventListener('click', showExperimentDetails);


    // Functions

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
                            <pre class="p-2 bg-dark rounded">${escapeHtml(iteration.system_prompt || '').substring(0, 200)}${iteration.system_prompt && iteration.system_prompt.length > 200 ? '...' : ''}</pre>
                        </div>

                        <div class="mb-3">
                            <h6>Output Prompt:</h6>
                            <pre class="p-2 bg-dark rounded">${escapeHtml(iteration.output_prompt || '').substring(0, 200)}${iteration.output_prompt && iteration.output_prompt.length > 200 ? '...' : ''}</pre>
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
                            <button class="btn btn-sm btn-outline-primary view-examples" data-iteration="${iteration.iteration}" onclick="window.loadExamplesForIteration(${iteration.iteration})">
                                <i class="fa-solid fa-magnifying-glass me-1"></i> View Examples
                            </button>
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

        // Add event listeners to view examples buttons
        document.querySelectorAll('.view-examples').forEach(btn => {
            btn.addEventListener('click', function() {
                // Scroll to examples section
                document.getElementById('examples-container').scrollIntoView({
                    behavior: 'smooth'
                });

                // Note: The actual function call is now handled by the onclick attribute
                // to ensure global scope access
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

        document.getElementById('original-system-prompt').textContent = original.system_prompt || '';
        document.getElementById('optimized-system-prompt').textContent = optimized.system_prompt || '';

        document.getElementById('original-output-prompt').textContent = original.output_prompt || '';
        document.getElementById('optimized-output-prompt').textContent = optimized.output_prompt || '';

        document.getElementById('original-score').textContent = `${((original.metrics?.avg_score || 0) * 100).toFixed(1)}%`;
        document.getElementById('optimized-score').textContent = `${((optimized.metrics?.avg_score || 0) * 100).toFixed(1)}%`;

        document.getElementById('optimizer-reasoning').textContent = optimized.reasoning || 'No reasoning available';

        // Set up load examples button
        const loadExamplesBtn = document.getElementById('load-examples-btn');
        loadExamplesBtn.onclick = () => loadExampleResults(original.iteration, optimized.iteration);

        // Reset examples container
        document.getElementById('compare-examples-container').innerHTML = `
            <div class="p-4 text-center text-muted">
                <p>Click "Load Examples" to see detailed results for each test case</p>
            </div>
        `;

        // Show comparison view
        experimentDetailsEl.style.display = 'none';
        compareViewEl.style.display = 'block';
    }

    function loadExampleResults(originalIterationNumber, optimizedIterationNumber) {
        showSpinner();
        const examplesContainer = document.getElementById('compare-examples-container');

        // Get examples for both iterations
        Promise.all([
            fetch(`/experiments/${currentExperimentId}/examples/${originalIterationNumber}`),
            fetch(`/experiments/${currentExperimentId}/examples/${optimizedIterationNumber}`)
        ])
        .then(responses => Promise.all(responses.map(r => r.json())))
        .then(([originalData, optimizedData]) => {
            if (originalData.error || optimizedData.error) {
                showAlert('Error loading examples', 'danger');
                return;
            }

            const originalExamples = originalData.examples || [];
            const optimizedExamples = optimizedData.examples || [];

            if (originalExamples.length === 0 && optimizedExamples.length === 0) {
                examplesContainer.innerHTML = `
                    <div class="p-4 text-center text-muted">
                        <p>No example results available for these iterations</p>
                    </div>
                `;
                return;
            }

            // Use optimized examples as the base list if available, otherwise use original
            const baseExamples = optimizedExamples.length > 0 ? optimizedExamples : originalExamples;

            // Clear container
            examplesContainer.innerHTML = '';

            // Create a map for quick lookup of original examples by user input
            const originalExamplesMap = {};
            originalExamples.forEach(ex => {
                originalExamplesMap[ex.user_input] = ex;
            });

            // Add filters and statistics summary
            const perfectOriginal = originalExamples.filter(ex => ex.score >= 0.9).length;
            const perfectOptimized = optimizedExamples.filter(ex => ex.score >= 0.9).length;
            const avgScoreOriginal = originalExamples.length > 0 ? originalExamples.reduce((sum, ex) => sum + (ex.score || 0), 0) / originalExamples.length : 0;
            const avgScoreOptimized = optimizedExamples.length > 0 ? optimizedExamples.reduce((sum, ex) => sum + (ex.score || 0), 0) / optimizedExamples.length : 0;

            const summaryEl = document.createElement('div');
            summaryEl.className = 'mb-4 p-3 bg-light border rounded';
            summaryEl.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h5>Original Iteration ${originalIterationNumber}</h5>
                        <div>Average Score: ${(avgScoreOriginal * 100).toFixed(1)}%</div>
                        <div>Perfect Matches: ${perfectOriginal}/${originalExamples.length}</div>
                    </div>
                    <div class="col-md-6">
                        <h5>Optimized Iteration ${optimizedIterationNumber}</h5>
                        <div>Average Score: ${(avgScoreOptimized * 100).toFixed(1)}%</div>
                        <div>Perfect Matches: ${perfectOptimized}/${optimizedExamples.length}</div>
                        <div class="mt-1">
                            <span class="badge ${avgScoreOptimized > avgScoreOriginal ? 'bg-success' : 'bg-danger'}">
                                ${avgScoreOptimized > avgScoreOriginal ? '+' : ''}${((avgScoreOptimized - avgScoreOriginal) * 100).toFixed(1)}% Overall
                            </span>
                        </div>
                    </div>
                </div>
                <div class="mt-3">
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary active filter-examples-btn" data-filter="all">All Examples</button>
                        <button class="btn btn-outline-primary filter-examples-btn" data-filter="improved">Improved</button>
                        <button class="btn btn-outline-primary filter-examples-btn" data-filter="perfect">Perfect Matches</button>
                        <button class="btn btn-outline-primary filter-examples-btn" data-filter="worse">Worse</button>
                    </div>
                </div>
            `;
            examplesContainer.appendChild(summaryEl);

            // Add filter functionality
            examplesContainer.querySelectorAll('.filter-examples-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    // Update active state
                    examplesContainer.querySelectorAll('.filter-examples-btn').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');

                    // Filter examples
                    const filter = this.getAttribute('data-filter');
                    const exampleItems = examplesContainer.querySelectorAll('.example-item');

                    exampleItems.forEach(item => {
                        const improvement = parseFloat(item.getAttribute('data-improvement'));
                        const optimizedScore = parseFloat(item.getAttribute('data-optimized-score'));

                        if (filter === 'all') {
                            item.style.display = '';
                        } else if (filter === 'improved' && improvement > 0) {
                            item.style.display = '';
                        } else if (filter === 'perfect' && optimizedScore >= 0.9) {
                            item.style.display = '';
                        } else if (filter === 'worse' && improvement < 0) {
                            item.style.display = '';
                        } else {
                            item.style.display = 'none';
                        }
                    });
                });
            });

            // Display examples
            baseExamples.forEach((example, index) => {
                const userInput = example.user_input;
                const groundTruth = example.ground_truth_output;
                const optimizedResponse = example.model_response;
                const optimizedScore = example.score;

                // Try to find matching original example
                const originalExample = originalExamplesMap[userInput];
                const originalResponse = originalExample ? originalExample.model_response : 'Not available';
                const originalScore = originalExample ? originalExample.score : 0;

                // Calculate score improvement
                const scoreImprovement = optimizedScore - originalScore;
                const scoreClass = scoreImprovement > 0 ? 'text-success' :
                                      scoreImprovement < 0 ? 'text-danger' : 'text-muted';
                const scoreSign = scoreImprovement > 0 ? '+' : '';

                // Create example card
                const exampleEl = document.createElement('div');
                exampleEl.className = 'accordion-item example-item';
                exampleEl.setAttribute('data-improvement', scoreImprovement);
                exampleEl.setAttribute('data-optimized-score', optimizedScore);
                exampleEl.innerHTML = `
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" 
                                data-bs-toggle="collapse" data-bs-target="#example-${index}">
                            <div class="d-flex justify-content-between align-items-center w-100 me-3">
                                <span><strong>Example ${index + 1}</strong></span>
                                <div>
                                    <span class="badge bg-secondary me-2">${(originalScore * 100).toFixed(1)}%</span>
                                    <i class="fa-solid fa-arrow-right mx-1"></i>
                                    <span class="badge bg-primary ms-2">${(optimizedScore * 100).toFixed(1)}%</span>
                                    <span class="badge ${scoreClass} ms-2">${scoreSign}${(scoreImprovement * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        </button>
                    </h2>
                    <div id="example-${index}" class="accordion-collapse collapse">
                        <div class="accordion-body">
                            <div class="mb-3">
                                <h6 class="d-flex align-items-center">
                                    <i class="fa-solid fa-keyboard me-2"></i> User Input:
                                </h6>
                                <div class="p-3 bg-light border rounded">
                                    ${escapeHtml(userInput)}
                                </div>
                            </div>

                            <div class="mb-3">
                                <h6 class="d-flex align-items-center">
                                    <i class="fa-solid fa-check-circle me-2"></i> Ground Truth Output:
                                </h6>
                                <div class="p-3 bg-light border rounded">
                                    ${escapeHtml(groundTruth)}
                                </div>
                            </div>

                            <div class="row g-3 mb-3">
                                <div class="col-md-6">
                                    <h6 class="d-flex align-items-center">
                                        <i class="fa-solid fa-robot me-2"></i> Original Response: 
                                        <span class="badge bg-secondary ms-2">${(originalScore * 100).toFixed(1)}%</span>
                                    </h6>
                                    <div class="p-3 bg-light border rounded" style="max-height: 300px; overflow-y: auto;">
                                        ${escapeHtml(originalResponse)}
                                    </div>
                                </div>

                                <div class="col-md-6">
                                    <h6 class="d-flex align-items-center">
                                        <i class="fa-solid fa-robot me-2"></i> Optimized Response: 
                                        <span class="badge bg-primary ms-2">${(optimizedScore * 100).toFixed(1)}%</span>
                                    </h6>
                                    <div class="p-3 border rounded ${optimizedScore >= 0.9 ? 'border-success' : ''} rounded" style="max-height: 300px; overflow-y: auto;">
                                        ${escapeHtml(optimizedResponse)}
                                    </div>
                                </div>
                            </div>

                            <div class="mt-3">
                                <h6 class="d-flex align-items-center">
                                    <i class="fa-solid fa-calculator me-2"></i> Score Calculation
                                </h6>
                                <div class="p-3 bg-light border rounded">
                                    <p class="mb-1"><strong>Improvement: </strong> <span class="${scoreClass}">${scoreSign}${(scoreImprovement * 100).toFixed(1)}%</span></p>
                                    <p class="mb-0">
                                        Scores are calculated based on semantic similarity to the ground truth, exact matches of key phrases, and correct structure.
                                        A score ≥ 90% is considered a perfect match.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                examplesContainer.appendChild(exampleEl);
            });
        })
        .catch(error => {
            console.error('Error loading example results:', error);
            showAlert('Error loading example results', 'danger');
            examplesContainer.innerHTML = `
                <div class="p-4 text-center text-danger">
                    <p>Error loading examples: ${error.message}</p>
                </div>
            `;
        })
        .finally(() => {
            hideSpinner();
        });
    }

    // Examples handling
    // Make sure loadExamplesForIteration is available globally
    window.loadExamplesForIteration = function(iteration) {
        console.log(`Loading examples for iteration: ${iteration}`);
        const examplesContainer = document.getElementById('examples-container');
        // Check if examplesContainer exists
        if (!examplesContainer) {
            console.error('Examples container not found');
            showAlert('Error: Examples container not found', 'danger');
            return;
        }

        const examplesLoading = document.getElementById('examples-loading');
        const noExamplesMessage = document.getElementById('no-examples-message');

        if (!examplesLoading || !noExamplesMessage) {
            console.error('Examples loading elements not found');
            showAlert('Error: Examples loading elements not found', 'danger');
            return;
        }

        // Reset container and show loading
        const examplesList = Array.from(examplesContainer.querySelectorAll('.example-card'));
        examplesList.forEach(el => el.remove());
        examplesLoading.style.display = 'block';
        noExamplesMessage.style.display = 'none';

        if (!currentExperimentData || !currentExperimentData.iterations) {
            console.error('Current experiment data not available');
            examplesLoading.style.display = 'none';
            noExamplesMessage.style.display = 'block';
            showAlert('Error: No experiment data available', 'danger');
            return;
        }

        // Find the iteration data from the currently loaded experiment
        const iterationData = currentExperimentData.iterations.find(it => it.iteration === iteration);

        if (!iterationData || !iterationData.examples) {
            // Need to fetch examples from the server
            fetch(`/experiments/${currentExperimentId}/iterations/${iteration}/examples`)
                .then(response => response.json())
                .then(data => {
                    examplesLoading.style.display = 'none';

                    if (data.error) {
                        showAlert(data.error, 'danger');
                        noExamplesMessage.style.display = 'block';
                    } else if (data.examples && data.examples.length > 0) {
                        // Store examples in the iteration data for reuse
                        const currentIteration = currentExperimentData.iterations.find(it => it.iteration === iteration);
                        if (currentIteration) {
                            currentIteration.examples = data.examples;
                        }

                        // Display examples
                        renderExamples(data.examples);
                    } else {
                        noExamplesMessage.style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error('Error loading examples:', error);
                    examplesLoading.style.display = 'none';
                    noExamplesMessage.style.display = 'block';
                    showAlert('Error loading examples', 'danger');
                });
        } else {
            // We already have the examples data
            examplesLoading.style.display = 'none';

            if (iterationData.examples.length > 0) {
                renderExamples(iterationData.examples);
            } else {
                noExamplesMessage.style.display = 'block';
            }
        }
    }

    function renderExamples(examples) {
        const examplesContainer = document.getElementById('examples-container');
        if (!examplesContainer) {
            console.error('Examples container not found');
            return;
        }

        // Clear any existing examples
        const examplesList = Array.from(examplesContainer.querySelectorAll('.example-card'));
        examplesList.forEach(el => el.remove());

        // Get current filter
        const activeFilter = document.querySelector('.filter-examples.active');
        if (!activeFilter) {
            console.error('Active filter not found');
            return;
        }

        const activeFilterValue = activeFilter.getAttribute('data-filter');

        // Sort examples by score (highest first)
        examples.sort((a, b) => (b.score || 0) - (a.score || 0));

        // Apply filter
        let filteredExamples = examples;
        if (activeFilterValue === 'perfect') {
            filteredExamples = examples.filter(ex => ex.score >= 0.9);
        } else if (activeFilterValue === 'imperfect') {
            filteredExamples = examples.filter(ex => ex.score < 0.9);
        }

        if (filteredExamples.length === 0) {
            // No examples match the filter
            const noMatchMessage = document.createElement('div');
            noMatchMessage.className = 'text-center py-4 text-muted';
            noMatchMessage.innerHTML = `
                <i class="fa-solid fa-filter-circle-xmark me-2"></i>
                No examples match the current filter
            `;
            examplesContainer.appendChild(noMatchMessage);
            return;
        }

        // Show metrics summary
        const perfectMatches = examples.filter(ex => ex.score >= 0.9).length;
        const avgScore = examples.length > 0 ? examples.reduce((sum, ex) => sum + (ex.score || 0), 0) / examples.length : 0;

        const summaryCard = document.createElement('div');
        summaryCard.className = 'card mb-4';
        summaryCard.innerHTML = `
            <div class="card-body p-3">
                <h5 class="card-title">Examples Summary - Iteration ${currentExperimentData.iterations.find(it => it.iteration === iteration)?.iteration || iteration}</h5>
                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="d-flex align-items-center">
                            <div class="fs-4 me-2">${(avgScore * 100).toFixed(1)}%</div>
                            <div class="text-muted">Average Score</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="d-flex align-items-center">
                            <div class="fs-4 me-2">${perfectMatches}</div>
                            <div class="text-muted">Perfect Matches</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="d-flex align-items-center">
                            <div class="fs-4 me-2">${examples.length}</div>
                            <div class="text-muted">Total Examples</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        examplesContainer.appendChild(summaryCard);

        // Render examples
        filteredExamples.forEach((example, index) => {
            const score = example.score || 0;
            const isPerfectMatch = score >= 0.9;

            const exampleCard = document.createElement('div');
            exampleCard.className = 'example-card border rounded mb-3';
            exampleCard.setAttribute('data-score', score);

            // Determine score badge color based on score
            let badgeClass = 'bg-danger';
            if (score >= 0.9) {
                badgeClass = 'bg-success';
            } else if (score >= 0.7) {
                badgeClass = 'bg-warning text-dark';
            } else if (score >= 0.5) {
                badgeClass = 'bg-info';
            }

            exampleCard.innerHTML = `
                <div class="p-3">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h6 class="mb-0 d-flex align-items-center">
                            <i class="fa-solid fa-clipboard-check me-2"></i>
                            Example ${index + 1}
                        </h6>
                        <span class="badge ${badgeClass}">${(score * 100).toFixed(1)}%</span>
                    </div>

                    <div class="mb-3">
                        <div class="fw-bold text-muted small mb-1 d-flex align-items-center">
                            <i class="fa-solid fa-keyboard me-2"></i> USER INPUT:
                        </div>
                        <div class="p-2 bg-light border rounded user-input" style="max-height: 150px; overflow-y: auto;">
                            ${escapeHtml(example.user_input)}
                        </div>
                    </div>

                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="fw-bold text-muted small mb-1 d-flex align-items-center">
                                <i class="fa-solid fa-check-circle me-2"></i> EXPECTED OUTPUT:
                            </div>
                            <div class="p-2 bg-light border rounded expected-output" style="max-height: 200px; overflow-y: auto;">
                                ${escapeHtml(example.ground_truth_output)}
                            </div>
                        </div>

                        <div class="col-md-6">
                            <div class="fw-bold text-muted small mb-1 d-flex align-items-center">
                                <i class="fa-solid fa-robot me-2"></i> MODEL RESPONSE:
                            </div>
                            <div class="p-2 border rounded model-response ${isPerfectMatch ? 'border-success bg-success bg-opacity-10' : 'border-danger bg-danger bg-opacity-10'}" style="max-height: 200px; overflow-y: auto;">
                                ${escapeHtml(example.model_response)}
                            </div>
                        </div>
                    </div>

                    <div class="mt-3">
                        <div class="fw-bold text-muted small mb-1 d-flex align-items-center">
                            <i class="fa-solid fa-calculator me-2"></i> HOW SCORE IS CALCULATED:
                        </div>
                        <div class="p-2 bg-light border rounded small">
                            <p class="mb-1">The score is calculated using:</p>
                            <ul class="mb-1">
                                <li>Sequence similarity (70%): How similar the text sequences are</li>
                                <li>Keyword matching (30%): How many important keywords match</li>
                            </ul>
                            <p class="mb-0">A score ≥ 90% is considered a perfect match.</p>
                        </div>
                    </div>
                </div>
            `;

            examplesContainer.appendChild(exampleCard);
        });
    }

    // Set up filter buttons for examples
    document.querySelectorAll('.filter-examples').forEach(btn => {
        btn.addEventListener('click', function() {
            // Update active state
            document.querySelectorAll('.filter-examples').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // Get currently loaded examples
            const activeIteration = document.querySelector('.accordion-collapse.show');
            if (activeIteration) {
                const iterationId = activeIteration.id.replace('iteration-', '');
                if (currentExperimentData && currentExperimentData.iterations && iterationId) {
                    const iterationData = currentExperimentData.iterations[iterationId];
                    if (iterationData && iterationData.examples) {
                        renderExamples(iterationData.examples);
                    }
                }
            } else if (currentExperimentData && currentExperimentData.iterations && 
                      currentExperimentData.iterations.length > 0 &&
                      currentExperimentData.iterations[currentExperimentData.iterations.length - 1].examples) {
                // If no iteration is open, but we have examples, render the last iteration's examples
                renderExamples(currentExperimentData.iterations[currentExperimentData.iterations.length - 1].examples);
            }
        });
    });

    function setupHistoryChart() {
        const ctx = document.getElementById('history-chart');
        if (!ctx) {
            console.error('History chart canvas not found');
            return;
        }

        try {
        historyChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Average Score',
                        data: [],
                        borderColor: 'rgba(75, 192, 192,1)',
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
        } catch (error) {
            console.error('Error initializing chart:', error);
            // Fall back to a simple text display if chart fails
            const chartContainer = ctx.parentElement;
            if (chartContainer) {
                chartContainer.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fa-solid fa-triangle-exclamation me-2"></i>
                        Chart rendering failed. Please try refreshing the page.
                    </div>
                    <div class="p-3 bg-light rounded">
                        <h5>Performance Metrics Summary</h5>
                        <div id="text-metrics-summary">No data available</div>
                    </div>
                `;
            }
        }
    }

    function updateHistoryChart(iterations) {
        if (!historyChart) {
            // Update text summary if chart failed to initialize
            const textSummary = document.getElementById('text-metrics-summary');
            if (textSummary) {
                let summaryHTML = '<ul>';
                iterations.forEach(item => {
                    summaryHTML += `<li>Iteration ${item.iteration}: Score ${((item.metrics?.avg_score || 0) * 100).toFixed(1)}%, Perfect Matches ${(item.metrics?.perfect_match_percent || 0).toFixed(1)}%</li>`;
                });
                summaryHTML += '</ul>';
                textSummary.innerHTML = summaryHTML;
            }
            return;
        }

        try {
            const labels = iterations.map(item => `Iteration ${item.iteration}`);
            const avgScores = iterations.map(item => (item.metrics?.avg_score || 0) * 100);
            const perfectMatches = iterations.map(item => item.metrics?.perfect_match_percent || 0);

            historyChart.data.labels = labels;
            historyChart.data.datasets[0].data = avgScores;
            historyChart.data.datasets[1].data = perfectMatches;
            historyChart.update();
        } catch (error) {
            console.error('Error updating chart:', error);
            showAlert('Error updating performance chart', 'warning');
        }
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
        if (!alertContainer) {
            console.error('Alert container not found');
            return;
        }

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        alertContainer.appendChild(alert);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            try {
                const alertInstance = bootstrap.Alert.getOrCreateInstance(alert);
                alertInstance.close();
            } catch (error) {
                console.error('Error auto-dismissing alert:', error);
                alert.remove();
            }
        }, 5000);
    }

    function escapeHtml(text) {
        if (!text) return '';
        try {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        } catch (error) {
            console.error('Error escaping HTML:', error);
            return '';
        }
    }
});