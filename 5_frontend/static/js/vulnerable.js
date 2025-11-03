document.addEventListener('DOMContentLoaded', () => {
    
    const loginForm = document.getElementById('vulnerable-login-form');
    const searchForm = document.getElementById('vulnerable-search-form');
    const apiResponseEl = document.getElementById('api-response');
    const loginResultEl = document.getElementById('login-result-container');
    const searchResultEl = document.getElementById('search-result-container');

    // --- Vulnerable Login ---
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const username = document.getElementById('vuln-login-user').value;
            const password = document.getElementById('vuln-login-pass').value;

            fetch('/api/vulnerable/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
            })
            .then(response => response.json())
            .then(data => {
                apiResponseEl.textContent = JSON.stringify(data, null, 2);
                if (data.success) {
                    loginResultEl.innerHTML = `<div class="alert alert-success">Login successful! Welcome, ${escapeHtml(data.user.username)}.</div>`;
                } else {
                    loginResultEl.innerHTML = `<div class="alert alert-danger">Login Failed: ${escapeHtml(data.message)}</div>`;
                }
            })
            .catch(err => {
                apiResponseEl.textContent = `Error: ${err}`;
                loginResultEl.innerHTML = `<div class="alert alert-danger">Client-side error.</div>`;
            });
        });
    }

    // --- Vulnerable Search ---
    if (searchForm) {
        searchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const searchTerm = document.getElementById('vuln-search-term').value;

            fetch(`/api/vulnerable/search?q=${encodeURIComponent(searchTerm)}`)
            .then(response => response.json())
            .then(data => {
                apiResponseEl.textContent = JSON.stringify(data, null, 2);
                
                let resultHtml = '';
                if (data.success && data.results.length > 0) {
                    resultHtml = '<ul class="list-group">';
                    data.results.forEach(item => {
                        // In a real vulnerability, this is where an XSS payload might fire
                        // We use escapeHtml to show the data safely
                        resultHtml += `<li class="list-group-item"><strong>Name:</strong> ${escapeHtml(item.name)} | <strong>Desc:</strong> ${escapeHtml(item.description)}</li>`;
                    });
                    resultHtml += '</ul>';
                    searchResultEl.innerHTML = resultHtml;
                } else if (data.success) {
                    searchResultEl.innerHTML = `<div class="alert alert-info">No results found.</div>`;
                } else {
                    searchResultEl.innerHTML = `<div class="alert alert-danger">Search Failed: ${escapeHtml(data.message)}</div>`;
                }
            })
            .catch(err => {
                apiResponseEl.textContent = `Error: ${err}`;
                searchResultEl.innerHTML = `<div class="alert alert-danger">Client-side error.</div>`;
            });
        });
    }

});