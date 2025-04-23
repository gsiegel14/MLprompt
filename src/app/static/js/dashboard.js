
// Dashboard JavaScript

// Format numbers with commas
function formatNumber(number) {
    return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Add number formatting filter for Jinja templates
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Format any elements with the 'format-number' class
    document.querySelectorAll('.format-number').forEach(function(element) {
        var value = parseInt(element.textContent);
        if (!isNaN(value)) {
            element.textContent = formatNumber(value);
        }
    });

    // Add event listeners for date range selectors if they exist
    var dateRangeSelectors = document.querySelectorAll('.date-range-selector');
    if (dateRangeSelectors.length > 0) {
        dateRangeSelectors.forEach(function(selector) {
            selector.addEventListener('change', function() {
                window.location.href = this.value;
            });
        });
    }

    // Add refresh data button functionality if it exists
    var refreshButton = document.getElementById('refresh-data');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            location.reload();
        });
    }
});

// Auto-refresh dashboard if enabled
function setupAutoRefresh() {
    var refreshElement = document.getElementById('auto-refresh');
    if (refreshElement && refreshElement.dataset.enabled === 'true') {
        var refreshInterval = parseInt(refreshElement.dataset.interval) || 30000; // Default to 30 seconds
        setInterval(function() {
            location.reload();
        }, refreshInterval);
    }
}

// Call setupAutoRefresh when the document is loaded
document.addEventListener('DOMContentLoaded', setupAutoRefresh);
