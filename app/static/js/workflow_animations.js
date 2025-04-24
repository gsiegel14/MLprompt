/**
 * Workflow Animations JS
 * 
 * Provides animated progress indicators for the 5-API workflow steps
 */

// Class to manage workflow animations
class WorkflowAnimator {
    constructor(containerId = 'workflow-container') {
        this.container = document.getElementById(containerId);
        this.steps = document.querySelectorAll('.workflow-step');
        this.connectors = document.querySelectorAll('.workflow-connector');
        this.activeClass = 'active-step';
        this.completeClass = 'complete-step';
        this.processingClass = 'processing-step';
        this.currentStep = 0;
        this.isAnimating = false;
        this.animationQueue = [];
    }

    /**
     * Initialize the workflow animation
     */
    init() {
        // Reset any previous state
        this.resetWorkflow();
        
        // Add animation classes to steps
        this.steps.forEach(step => {
            step.querySelector('.step-circle').innerHTML += '<div class="spinner-border spinner-border-sm text-light d-none" role="status"><span class="visually-hidden">Loading...</span></div>';
        });
        
        // Add pulse animation to connectors
        this.connectors.forEach(connector => {
            connector.innerHTML = '<div class="connector-progress"></div>';
        });
    }

    /**
     * Reset workflow to initial state
     */
    resetWorkflow() {
        this.currentStep = 0;
        this.steps.forEach(step => {
            step.classList.remove(this.activeClass, this.completeClass, this.processingClass);
        });
        this.connectors.forEach(connector => {
            connector.classList.remove('active-connector', 'complete-connector');
            const progress = connector.querySelector('.connector-progress');
            if (progress) progress.style.width = '0%';
        });
    }

    /**
     * Activate a specific step
     * @param {number} stepIndex - Index of the step to activate (0-based)
     * @param {boolean} instant - If true, activate instantly without animation
     */
    activateStep(stepIndex, instant = false) {
        if (stepIndex < 0 || stepIndex >= this.steps.length) return;
        
        const step = this.steps[stepIndex];
        
        // If instant, just apply the active state
        if (instant) {
            step.classList.add(this.activeClass);
            this.currentStep = stepIndex;
            return;
        }
        
        // Show processing state first
        step.classList.add(this.processingClass);
        
        // Show spinner
        const spinner = step.querySelector('.spinner-border');
        if (spinner) spinner.classList.remove('d-none');
        
        // Set active after a short delay
        setTimeout(() => {
            step.classList.remove(this.processingClass);
            step.classList.add(this.activeClass);
            const spinner = step.querySelector('.spinner-border');
            if (spinner) spinner.classList.add('d-none');
            this.currentStep = stepIndex;
            
            // Animate the connector to the next step if there is one
            if (stepIndex < this.connectors.length) {
                this.animateConnector(stepIndex);
            }
        }, 1500); // Processing animation duration
    }

    /**
     * Complete a step (mark as done)
     * @param {number} stepIndex - Index of the step to complete
     */
    completeStep(stepIndex) {
        if (stepIndex < 0 || stepIndex >= this.steps.length) return;
        
        const step = this.steps[stepIndex];
        step.classList.remove(this.activeClass, this.processingClass);
        step.classList.add(this.completeClass);
        
        // Check if there's a connector after this step
        if (stepIndex < this.connectors.length) {
            this.completeConnector(stepIndex);
        }
    }

    /**
     * Animate connector progress
     * @param {number} connectorIndex - Index of the connector to animate
     */
    animateConnector(connectorIndex) {
        if (connectorIndex < 0 || connectorIndex >= this.connectors.length) return;
        
        const connector = this.connectors[connectorIndex];
        connector.classList.add('active-connector');
        
        const progress = connector.querySelector('.connector-progress');
        if (progress) {
            progress.style.width = '0%';
            
            // Animate the progress
            let width = 0;
            const interval = setInterval(() => {
                if (width >= 100) {
                    clearInterval(interval);
                    // Optionally, trigger the next step here
                } else {
                    width += 1;
                    progress.style.width = width + '%';
                }
            }, 20); // 20ms interval gives a 2-second animation for 0-100%
        }
    }

    /**
     * Complete a connector (mark as done)
     * @param {number} connectorIndex - Index of the connector to complete
     */
    completeConnector(connectorIndex) {
        if (connectorIndex < 0 || connectorIndex >= this.connectors.length) return;
        
        const connector = this.connectors[connectorIndex];
        connector.classList.remove('active-connector');
        connector.classList.add('complete-connector');
        
        const progress = connector.querySelector('.connector-progress');
        if (progress) progress.style.width = '100%';
    }

    /**
     * Progress the workflow to the next step
     */
    nextStep() {
        if (this.currentStep < this.steps.length - 1) {
            if (this.currentStep >= 0) {
                this.completeStep(this.currentStep);
            }
            
            this.activateStep(this.currentStep + 1);
            return true;
        }
        return false;
    }

    /**
     * Run a full workflow animation demo
     */
    runDemo() {
        this.resetWorkflow();
        
        // Activate first step immediately
        this.activateStep(0, true);
        
        // Schedule the progression of steps
        let stepDelay = 500; // Start delay
        
        for (let i = 0; i < this.steps.length; i++) {
            // Steps after the first need processing and connector animations
            if (i > 0) {
                setTimeout(() => {
                    this.nextStep();
                }, stepDelay);
                stepDelay += 3000; // 3 second interval between steps
            }
        }
        
        // Complete the final step
        setTimeout(() => {
            this.completeStep(this.steps.length - 1);
        }, stepDelay);
    }
}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add workflow animation styles
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .workflow-step {
            transition: all 0.3s ease-in-out;
        }
        
        .workflow-step .step-circle {
            position: relative;
            transition: all 0.3s ease-in-out;
        }
        
        .workflow-step .spinner-border {
            position: absolute;
            width: 20px;
            height: 20px;
            top: 50%;
            left: 50%;
            margin-top: -10px;
            margin-left: -10px;
        }
        
        .processing-step .step-circle {
            animation: pulse 1.5s infinite;
        }
        
        .active-step .step-circle {
            transform: scale(1.1);
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.3);
        }
        
        .complete-step .step-circle {
            background-color: #43a047 !important;
        }
        
        .complete-step .step-circle::after {
            content: '\\2713';  /* Checkmark symbol */
            position: absolute;
            font-size: 1.5rem;
        }
        
        .workflow-connector {
            overflow: hidden;
            transition: all 0.3s ease-in-out;
        }
        
        .connector-progress {
            height: 100%;
            width: 0%;
            background-color: #43a047;
            transition: width 0.1s linear;
        }
        
        .active-connector {
            background-color: transparent;
        }
        
        .complete-connector {
            background-color: transparent;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    `;
    document.head.appendChild(styleElement);
    
    // Initialize the workflow animator
    const workflowDiagrams = document.querySelectorAll('.workflow-diagram');
    
    workflowDiagrams.forEach((diagram, index) => {
        if (diagram) {
            diagram.id = `workflow-diagram-${index}`;
            const animator = new WorkflowAnimator(diagram.id);
            animator.init();
            
            // Add demo buttons if they don't exist
            if (!document.querySelector(`#workflow-demo-btn-${index}`)) {
                const demoBtn = document.createElement('button');
                demoBtn.id = `workflow-demo-btn-${index}`;
                demoBtn.className = 'btn btn-sm btn-outline-primary mt-3';
                demoBtn.innerHTML = '<i class="fa-solid fa-play me-1"></i> Run Animation Demo';
                demoBtn.addEventListener('click', () => animator.runDemo());
                
                // Add after diagram but before description
                const nextElement = diagram.nextElementSibling;
                if (nextElement) {
                    diagram.parentNode.insertBefore(demoBtn, nextElement);
                } else {
                    diagram.parentNode.appendChild(demoBtn);
                }
            }
            
            // Store the animator on the window for access from other scripts
            window[`workflowAnimator${index}`] = animator;
        }
    });
    
    // If there's a 5-API workflow page, add controls
    const apiWorkflowPage = document.getElementById('five-api-workflow-page');
    if (apiWorkflowPage) {
        const simulateBtn = document.getElementById('simulate-workflow-btn');
        if (simulateBtn && window.workflowAnimator0) {
            simulateBtn.addEventListener('click', () => window.workflowAnimator0.runDemo());
        }
    }
});

// Export the class for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WorkflowAnimator };
}