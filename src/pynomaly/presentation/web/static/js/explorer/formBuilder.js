document.addEventListener('htmx:configRequest', (event) => {
    // Example of setting headers for HTMX requests
    event.detail.headers['X-CSRFToken'] = 'fetchCSRFToken()';
});

document.addEventListener('DOMContentLoaded', () => {
    const contentArea = document.querySelector('.content');

    contentArea.innerHTML = `
        <form hx-post='/api/v1/detection/predict' hx-target='.response'>
            <label for='dataset'>Dataset:</label>
            <input type='text' name='dataset' required />

            <label for='detector'>Detector ID:</label>
            <input type='text' name='detector' required />

            <button type='submit'>Run Detection</button>
        </form>
        <div class='response'></div>
    `;
});
