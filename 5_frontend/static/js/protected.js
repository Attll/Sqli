document.addEventListener('DOMContentLoaded', () => {

    const loginForm = document.getElementById('protected-login-form');
    const searchForm = document.getElementById('protected-search-form');
    const apiResponseEl = document.getElementById('api-response');
    const responseContainer = document.getElementById('api-response-container');

    // --- Protected Login ---
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const username = document.getElementById('prot-login-user').value;
            const password = document.getElementById('prot-login-pass').value;

            fetch('/api/protected/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
            })
            .then(response => response.json())
            .then(data => {
                // Show detection result if it was blocked
                if (data.detection && data.detection.is_malicious) {
                    showDetectionResult(data.detection, 'api-response-container');
                } else {
                    // Show the raw JSON response
                    responseContainer.innerHTML = `<pre id="api-response" class="bg-light p-3 rounded query-display">${JSON.stringify(data, null, 2)}</pre>`;
                }
            })
            .catch(err => {
                responseContainer.innerHTML = `<pre id="api-response" class="bg-light p-3 rounded query-display">Error: ${err}</pre>`;
            });
        });
    }

    // --- Protected Search ---
    if (searchForm) {
        searchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const searchTerm = document.getElementById('prot-search-term').value;

            fetch(`/api/protected/search?q=${encodeURIComponent(searchTerm)}`)
            .then(response => response.json())
            .then(data => {
                // Show detection result if it was blocked
                if (data.detection && data.detection.is_malicious) {
                    showDetectionResult(data.detection, 'api-response-container');
                } else {
                     // Show the raw JSON response
                    responseContainer.innerHTML = `<pre id="api-response" class="bg-light p-3 rounded query-display">${JSON.stringify(data, null, 2)}</pre>`;
                }
            })
            .catch(err => {
                responseContainer.innerHTML = `<pre id="api-response" class="bg-light p-3 rounded query-display">Error: ${err}</pre>`;
            });
        });
    }
});