// Common functions for all pages

/**
 * Sends a query to the detection API.
 * @param {string} query The query string to test.
 * @returns {Promise<object>} The detection result from the API.
 */
function detectSQLi(query) {
    return fetch('/api/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json());
}

/**
 * Displays a formatted detection result in a specified element.
 * @param {object} result The detection result object from the API.
 * @param {string} elementId The ID of the container element.
 */
function showDetectionResult(result, elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const className = result.is_malicious ? 'detection-blocked' : 'detection-allowed';
    const icon = result.is_malicious ? '🚨' : '✅';
    const status = result.is_malicious ? 'BLOCKED' : 'ALLOWED';

    element.className = `detection-result ${className}`;
    element.innerHTML = `
        <h4>${icon} ${status}</h4>
        <p><strong>Query:</strong> <code class="text-dark">${escapeHtml(result.query)}</code></p>
        <p><strong>Confidence (Malicious):</strong> ${(result.confidence * 100).toFixed(2)}%</p>
        <p><strong>Detection Time:</strong> ${result.detection_time_ms.toFixed(2)}ms</p>
        <div class="confidence-bar">
            <div class="confidence-indicator" style="left: ${result.confidence * 100}%"></div>
        </div>
    `;
}

/**
 * Escapes HTML special characters to prevent XSS.
 * @param {string} text The string to escape.
 * @returns {string} The escaped string.
 */
function escapeHtml(text) {
    if (text === null || text === undefined) {
        return '';
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}