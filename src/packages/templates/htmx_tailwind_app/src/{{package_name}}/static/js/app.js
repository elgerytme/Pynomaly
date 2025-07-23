// {{package_name}} JavaScript utilities

// Initialize HTMX
document.addEventListener('DOMContentLoaded', function() {
    // Configure HTMX
    htmx.config.defaultSwapStyle = "innerHTML";
    htmx.config.defaultSwapDelay = 0;
    htmx.config.defaultSettleDelay = 20;
    
    // Log HTMX events in development
    if (window.location.hostname === 'localhost') {
        htmx.logger = function(elt, event, data) {
            if(console) {
                console.log(event, elt, data);
            }
        }
    }
});

// Handle HTMX errors globally
document.body.addEventListener('htmx:responseError', function(evt) {
    console.error('HTMX Error:', evt.detail);
    
    // Show error notification
    htmx.ajax('GET', '/htmx/notifications/error?message=An error occurred. Please try again.', '#notifications');
});

// Handle HTMX before requests (show loading states)
document.body.addEventListener('htmx:beforeRequest', function(evt) {
    // Add loading class to target element
    const target = evt.detail.target;
    if (target) {
        target.classList.add('opacity-50', 'pointer-events-none');
    }
});

// Handle HTMX after requests (remove loading states)
document.body.addEventListener('htmx:afterRequest', function(evt) {
    // Remove loading class from target element
    const target = evt.detail.target;
    if (target) {
        target.classList.remove('opacity-50', 'pointer-events-none');
    }
});

// Utility functions
const app = {
    // Show notification
    showNotification: function(message, type = 'info') {
        htmx.ajax('GET', `/htmx/notifications/${type}?message=${encodeURIComponent(message)}`, '#notifications');
    },
    
    // Confirm action
    confirmAction: function(message) {
        return confirm(message);
    },
    
    // Format currency
    formatCurrency: function(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    },
    
    // Format date
    formatDate: function(date) {
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        }).format(new Date(date));
    }
};

// Export for use in templates
window.app = app;