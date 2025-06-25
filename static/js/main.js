// Main JavaScript functionality for CSV Data Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // File upload validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file size (10MB limit)
                // Check file size (100MB limit)
            if (file.size > 100 * 1024 * 1024) {
                alert('File size must be less than 100MB');
                e.target.value = '';
                return;
            }
                
                // Check file type
                if (!file.name.toLowerCase().endsWith('.csv')) {
                    alert('Please select a CSV file');
                    e.target.value = '';
                    return;
                }
                
                // Display file info
                displayFileInfo(file);
            }
        });
    });

    // Form validation helpers
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
            }
        });
    });
});

function displayFileInfo(file) {
    const fileSize = (file.size / 1024 / 1024).toFixed(2);
    const fileInfo = document.createElement('div');
    fileInfo.className = 'alert alert-info mt-2';
    fileInfo.innerHTML = `
        <i class="fas fa-file-csv me-2"></i>
        <strong>${file.name}</strong> (${fileSize} MB)
    `;
    
    // Remove existing file info
    const existingInfo = document.querySelector('.file-info');
    if (existingInfo) {
        existingInfo.remove();
    }
    
    // Add new file info
    fileInfo.classList.add('file-info');
    const fileInput = document.querySelector('input[type="file"]');
    fileInput.parentNode.appendChild(fileInfo);
}

function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'This field is required');
            isValid = false;
        } else {
            clearFieldError(field);
        }
    });
    
    return isValid;
}

function showFieldError(field, message) {
    clearFieldError(field);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'text-danger mt-1 field-error';
    errorDiv.textContent = message;
    
    field.parentNode.appendChild(errorDiv);
    field.classList.add('is-invalid');
}

function clearFieldError(field) {
    const existingError = field.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
    field.classList.remove('is-invalid');
}

// Utility functions for chart analysis
function updateChartOptions(xColumn, yColumn) {
    const chartTypeSelect = document.getElementById('chart-type');
    if (!chartTypeSelect) return;
    
    // Clear existing options
    chartTypeSelect.innerHTML = '<option value="">Select chart type</option>';
    
    // Add appropriate options based on column selection
    const options = getChartOptions(xColumn, yColumn);
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option.value;
        optionElement.textContent = option.text;
        chartTypeSelect.appendChild(optionElement);
    });
}

function getChartOptions(xColumn, yColumn) {
    const options = [
        { value: 'histogram', text: 'Histogram' },
        { value: 'bar', text: 'Bar Chart' },
        { value: 'pie', text: 'Pie Chart' },
        { value: 'box', text: 'Box Plot' }
    ];
    
    if (yColumn) {
        options.push(
            { value: 'scatter', text: 'Scatter Plot' },
            { value: 'line', text: 'Line Chart' }
        );
    }
    
    return options;
}

// Progress tracking
function updateProgress(step) {
    const steps = ['upload', 'columns', 'cleaning', 'analysis'];
    const currentIndex = steps.indexOf(step);
    
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        const percentage = ((currentIndex + 1) / steps.length) * 100;
        progressBar.style.width = percentage + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
    }
}

// Auto-save functionality for long forms
function enableAutoSave(formId) {
    const form = document.getElementById(formId);
    if (!form) return;
    
    const inputs = form.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('change', function() {
            saveFormData(formId);
        });
    });
    
    // Load saved data on page load
    loadFormData(formId);
}

function saveFormData(formId) {
    const form = document.getElementById(formId);
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    localStorage.setItem(`form_${formId}`, JSON.stringify(data));
}

function loadFormData(formId) {
    const savedData = localStorage.getItem(`form_${formId}`);
    if (!savedData) return;
    
    try {
        const data = JSON.parse(savedData);
        const form = document.getElementById(formId);
        
        Object.keys(data).forEach(key => {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = data[key];
            }
        });
    } catch (e) {
        console.error('Error loading saved form data:', e);
    }
}

// Clear saved form data
function clearSavedFormData(formId) {
    localStorage.removeItem(`form_${formId}`);
}

// Export functionality
function exportChart(chartId) {
    const chartImage = document.getElementById(chartId);
    if (!chartImage) return;
    
    const link = document.createElement('a');
    link.download = 'chart.png';
    link.href = chartImage.src;
    link.click();
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter to submit forms
    if (e.ctrlKey && e.key === 'Enter') {
        const activeForm = document.querySelector('form:focus-within');
        if (activeForm) {
            activeForm.submit();
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        const openModal = document.querySelector('.modal.show');
        if (openModal) {
            const modal = bootstrap.Modal.getInstance(openModal);
            if (modal) modal.hide();
        }
    }
});