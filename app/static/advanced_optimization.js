
document.addEventListener('DOMContentLoaded', function() {
  // Hyperparameter tuning form
  const hyperparamForm = document.getElementById('hyperparameterForm');
  const hyperparamResultsContainer = document.getElementById('hyperparamResultsContainer');
  const hyperparamSpinner = document.getElementById('hyperparamSpinner');
  const bestConfigResults = document.getElementById('bestConfigResults');
  const allConfigsResults = document.getElementById('allConfigsResults');
  const showAllConfigsBtn = document.getElementById('showAllConfigsBtn');
  const allConfigsContainer = document.getElementById('allConfigsContainer');
  const hyperparamSuccessAlert = document.getElementById('hyperparamSuccessAlert');
  
  // Cross-validation form
  const crossValidationForm = document.getElementById('crossValidationForm');
  const crossvalResultsContainer = document.getElementById('crossvalResultsContainer');
  const crossvalSpinner = document.getElementById('crossvalSpinner');
  const cvAvgScore = document.getElementById('cvAvgScore');
  const cvMinScore = document.getElementById('cvMinScore');
  const cvMaxScore = document.getElementById('cvMaxScore');
  const cvFoldDetails = document.getElementById('cvFoldDetails');
  const crossvalSuccessAlert = document.getElementById('crossvalSuccessAlert');
  
  // System comparison form
  const systemComparisonForm = document.getElementById('systemComparisonForm');
  const addSystemBtn = document.getElementById('addSystemBtn');
  const systemsContainer = document.getElementById('systemsContainer');
  const comparisonResultsContainer = document.getElementById('comparisonResultsContainer');
  const comparisonSpinner = document.getElementById('comparisonSpinner');
  const bestSystemName = document.getElementById('bestSystemName');
  const bestSystemScore = document.getElementById('bestSystemScore');
  const bestSystemDescription = document.getElementById('bestSystemDescription');
  const systemsComparisonTable = document.getElementById('systemsComparisonTable');
  const comparisonSuccessAlert = document.getElementById('comparisonSuccessAlert');
  
  // Initialize listeners
  initHyperparameterTuning();
  initCrossValidation();
  initSystemComparison();
  
  // Show/hide all configurations
  showAllConfigsBtn.addEventListener('click', function() {
    if (allConfigsContainer.style.display === 'none') {
      allConfigsContainer.style.display = 'block';
      showAllConfigsBtn.textContent = 'Hide All Configurations';
    } else {
      allConfigsContainer.style.display = 'none';
      showAllConfigsBtn.textContent = 'Show All Configurations';
    }
  });
  
  function initHyperparameterTuning() {
    hyperparamForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Show spinner
      hyperparamSpinner.classList.remove('d-none');
      
      // Get form values
      const systemPrompt = document.getElementById('systemPrompt').value;
      const outputPrompt = document.getElementById('outputPrompt').value;
      const temperatureValues = document.getElementById('temperatureValues').value.split(',').map(v => parseFloat(v.trim()));
      const strategyValues = document.getElementById('strategyValues').value.split(',').map(v => v.trim());
      const batchSizeValues = document.getElementById('batchSizeValues').value.split(',').map(v => parseInt(v.trim()));
      const iterationValues = document.getElementById('iterationValues').value.split(',').map(v => parseInt(v.trim()));
      const metricKey = document.getElementById('metricKey').value;
      
      // Validate inputs
      if (!systemPrompt || !outputPrompt) {
        alert('System and output prompts are required');
        hyperparamSpinner.classList.add('d-none');
        return;
      }
      
      // Prepare search space
      const searchSpace = {
        temperature: temperatureValues,
        optimizer_strategy: strategyValues,
        batch_size: batchSizeValues,
        max_iterations: iterationValues
      };
      
      // Prepare request data
      const requestData = {
        system_prompt: systemPrompt,
        output_prompt: outputPrompt,
        search_space: searchSpace,
        metric_key: metricKey
      };
      
      // Send request
      fetch('/api/v1/optimization/hyperparameter-tune', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        // Hide spinner
        hyperparamSpinner.classList.add('d-none');
        
        // Display results
        hyperparamResultsContainer.style.display = 'block';
        hyperparamSuccessAlert.textContent = 'Hyperparameter tuning completed successfully!';
        
        // Show best configuration
        if (data.results && data.results.best_configuration) {
          bestConfigResults.textContent = JSON.stringify(data.results.best_configuration, null, 2);
          
          // Show all configurations if available
          if (data.results.search_space) {
            allConfigsResults.textContent = JSON.stringify(data.results, null, 2);
          }
        } else {
          bestConfigResults.textContent = 'No optimal configuration found.';
        }
        
        // Scroll to results
        hyperparamResultsContainer.scrollIntoView({ behavior: 'smooth' });
      })
      .catch(error => {
        console.error('Error:', error);
        hyperparamSpinner.classList.add('d-none');
        alert('Error running hyperparameter tuning: ' + error.message);
      });
    });
  }
  
  function initCrossValidation() {
    crossValidationForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Show spinner
      crossvalSpinner.classList.remove('d-none');
      
      // Get form values
      const systemPrompt = document.getElementById('cvSystemPrompt').value;
      const outputPrompt = document.getElementById('cvOutputPrompt').value;
      const foldCount = document.getElementById('foldCount').value;
      
      // Validate inputs
      if (!systemPrompt || !outputPrompt) {
        alert('System and output prompts are required');
        crossvalSpinner.classList.add('d-none');
        return;
      }
      
      // Prepare request data
      const requestData = {
        system_prompt: systemPrompt,
        output_prompt: outputPrompt,
        fold_count: parseInt(foldCount)
      };
      
      // Send request
      fetch('/api/v1/optimization/cross-validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        // Hide spinner
        crossvalSpinner.classList.add('d-none');
        
        // Display results
        crossvalResultsContainer.style.display = 'block';
        crossvalSuccessAlert.textContent = 'Cross-validation completed successfully!';
        
        if (data.results) {
          // Update summary metrics
          cvAvgScore.textContent = data.results.average_score ? data.results.average_score.toFixed(2) : '0.00';
          cvMinScore.textContent = data.results.min_score ? data.results.min_score.toFixed(2) : '0.00';
          cvMaxScore.textContent = data.results.max_score ? data.results.max_score.toFixed(2) : '0.00';
          
          // Build fold details table
          if (data.results.fold_metrics && data.results.fold_metrics.length > 0) {
            let tableHtml = `
              <table class="table table-striped">
                <thead>
                  <tr>
                    <th>Fold</th>
                    <th>Score</th>
                    <th>Validation Size</th>
                    <th>Training Size</th>
                  </tr>
                </thead>
                <tbody>
            `;
            
            data.results.fold_metrics.forEach(fold => {
              tableHtml += `
                <tr>
                  <td>${fold.fold}</td>
                  <td>${fold.score.toFixed(2)}</td>
                  <td>${fold.validation_size}</td>
                  <td>${fold.training_size}</td>
                </tr>
              `;
            });
            
            tableHtml += `
                </tbody>
              </table>
            `;
            
            cvFoldDetails.innerHTML = tableHtml;
          } else {
            cvFoldDetails.innerHTML = '<p>No fold metrics available.</p>';
          }
        } else {
          cvFoldDetails.innerHTML = '<p>No results available.</p>';
        }
        
        // Scroll to results
        crossvalResultsContainer.scrollIntoView({ behavior: 'smooth' });
      })
      .catch(error => {
        console.error('Error:', error);
        crossvalSpinner.classList.add('d-none');
        alert('Error running cross-validation: ' + error.message);
      });
    });
  }
  
  function initSystemComparison() {
    // Add system button
    addSystemBtn.addEventListener('click', function() {
      const systemCards = document.querySelectorAll('.system-card');
      const nextId = systemCards.length + 1;
      
      const newSystemHtml = `
        <div class="system-card card mb-3" data-system-id="${nextId}">
          <div class="card-header d-flex justify-content-between align-items-center">
            <h5>System ${nextId}</h5>
            <button type="button" class="btn btn-sm btn-danger remove-system-btn">Remove</button>
          </div>
          <div class="card-body">
            <div class="row mb-3">
              <div class="col-md-12">
                <label class="form-label">System Name</label>
                <input type="text" class="form-control system-name" value="System ${nextId}">
              </div>
            </div>
            <div class="row">
              <div class="col-md-6">
                <label class="form-label">System Prompt</label>
                <textarea class="form-control system-prompt" rows="4" placeholder="Enter system prompt..."></textarea>
              </div>
              <div class="col-md-6">
                <label class="form-label">Output Prompt</label>
                <textarea class="form-control output-prompt" rows="4" placeholder="Enter output prompt..."></textarea>
              </div>
            </div>
          </div>
        </div>
      `;
      
      systemsContainer.insertAdjacentHTML('beforeend', newSystemHtml);
      
      // Ensure all remove buttons are visible
      document.querySelectorAll('.remove-system-btn').forEach(btn => {
        if (document.querySelectorAll('.system-card').length > 1) {
          btn.style.display = 'block';
        }
      });
      
      // Update remove button listeners
      updateRemoveButtonListeners();
    });
    
    // Handle form submission
    systemComparisonForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Show spinner
      comparisonSpinner.classList.remove('d-none');
      
      // Get all systems
      const systems = [];
      document.querySelectorAll('.system-card').forEach(card => {
        const name = card.querySelector('.system-name').value;
        const systemPrompt = card.querySelector('.system-prompt').value;
        const outputPrompt = card.querySelector('.output-prompt').value;
        
        // Only add if both prompts are provided
        if (systemPrompt && outputPrompt) {
          systems.push({
            name: name,
            system_prompt: systemPrompt,
            output_prompt: outputPrompt
          });
        }
      });
      
      // Validate inputs
      if (systems.length < 2) {
        alert('At least two valid systems with complete prompts are required');
        comparisonSpinner.classList.add('d-none');
        return;
      }
      
      // Get fold count
      const foldCount = document.getElementById('comparisonFoldCount').value;
      
      // Prepare request data
      const requestData = {
        systems: systems,
        fold_count: parseInt(foldCount)
      };
      
      // Send request
      fetch('/api/v1/optimization/compare-systems', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        // Hide spinner
        comparisonSpinner.classList.add('d-none');
        
        // Display results
        comparisonResultsContainer.style.display = 'block';
        comparisonSuccessAlert.textContent = 'System comparison completed successfully!';
        
        if (data.results && data.results.best_system) {
          // Update best system information
          bestSystemName.textContent = data.results.best_system.name;
          bestSystemScore.textContent = data.results.best_system.average_score.toFixed(2);
          
          // Create description
          bestSystemDescription.textContent = `This system achieved the highest score across ${foldCount} folds of cross-validation, with an average score of ${data.results.best_system.average_score.toFixed(2)}.`;
          
          // Create comparison table
          let tableHtml = `
            <table class="table table-striped">
              <thead>
                <tr>
                  <th>System Name</th>
                  <th>Average Score</th>
                  <th>Performance</th>
                </tr>
              </thead>
              <tbody>
          `;
          
          // Sort systems by score
          const sortedSystems = [...data.results.systems].sort((a, b) => 
            b.results.average_score - a.results.average_score);
          
          sortedSystems.forEach(system => {
            const score = system.results.average_score.toFixed(2);
            const isBest = system.name === data.results.best_system.name;
            
            tableHtml += `
              <tr class="${isBest ? 'table-success' : ''}">
                <td>${system.name}</td>
                <td>${score}</td>
                <td>
                  <div class="progress">
                    <div class="progress-bar ${isBest ? 'bg-success' : ''}" role="progressbar" 
                         style="width: ${score * 100}%;" 
                         aria-valuenow="${score * 100}" aria-valuemin="0" aria-valuemax="100">
                      ${score}
                    </div>
                  </div>
                </td>
              </tr>
            `;
          });
          
          tableHtml += `
              </tbody>
            </table>
          `;
          
          systemsComparisonTable.innerHTML = tableHtml;
          
          // Create comparison chart
          renderComparisonChart(sortedSystems);
        } else {
          systemsComparisonTable.innerHTML = '<p>No comparison results available.</p>';
        }
        
        // Scroll to results
        comparisonResultsContainer.scrollIntoView({ behavior: 'smooth' });
      })
      .catch(error => {
        console.error('Error:', error);
        comparisonSpinner.classList.add('d-none');
        alert('Error comparing systems: ' + error.message);
      });
    });
    
    // Update remove button listeners initially
    updateRemoveButtonListeners();
  }
  
  function updateRemoveButtonListeners() {
    document.querySelectorAll('.remove-system-btn').forEach(btn => {
      // Remove old listeners
      const newBtn = btn.cloneNode(true);
      btn.parentNode.replaceChild(newBtn, btn);
      
      // Add new listener
      newBtn.addEventListener('click', function() {
        const card = this.closest('.system-card');
        card.remove();
        
        // Hide remove button if only one system remains
        if (document.querySelectorAll('.system-card').length <= 1) {
          document.querySelector('.remove-system-btn').style.display = 'none';
        }
        
        // Renumber remaining systems
        document.querySelectorAll('.system-card').forEach((card, index) => {
          const newId = index + 1;
          card.setAttribute('data-system-id', newId);
          card.querySelector('h5').textContent = `System ${newId}`;
          
          // Update system name if it matches the default pattern
          const nameInput = card.querySelector('.system-name');
          if (nameInput.value.match(/^System \d+$/)) {
            nameInput.value = `System ${newId}`;
          }
        });
      });
    });
    
    // Show/hide remove buttons based on system count
    if (document.querySelectorAll('.system-card').length <= 1) {
      document.querySelector('.remove-system-btn').style.display = 'none';
    } else {
      document.querySelectorAll('.remove-system-btn').forEach(btn => {
        btn.style.display = 'block';
      });
    }
  }
  
  function renderComparisonChart(systems) {
    const chartContainer = document.getElementById('systemsComparisonChart');
    
    // Get canvas element or create one
    let canvas = chartContainer.querySelector('canvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      chartContainer.appendChild(canvas);
    }
    
    // Prepare data
    const labels = systems.map(s => s.name);
    const scores = systems.map(s => s.results.average_score);
    
    // Create chart
    const ctx = canvas.getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Average Score',
          data: scores,
          backgroundColor: systems.map((s, i) => 
            i === 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(54, 162, 235, 0.7)'
          ),
          borderColor: systems.map((s, i) => 
            i === 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(54, 162, 235, 1)'
          ),
          borderWidth: 1
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
        }
      }
    });
  }
});
