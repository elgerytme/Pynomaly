// MathJax configuration for anomaly detection documentation

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      '[+]': ['ams', 'newcommand', 'configMacros']
    },
    macros: {
      // Common anomaly detection math macros
      R: "{\\mathbb{R}}",
      N: "{\\mathbb{N}}",
      Z: "{\\mathbb{Z}}",
      Q: "{\\mathbb{Q}}",
      C: "{\\mathbb{C}}",
      
      // Vectors and matrices
      vect: ["{\\boldsymbol{#1}}", 1],
      mat: ["{\\mathbf{#1}}", 1],
      
      // Probability and statistics
      prob: ["{\\text{P}\\left(#1\\right)}", 1],
      expect: ["{\\text{E}\\left[#1\\right]}", 1],
      var: ["{\\text{Var}\\left(#1\\right)}", 1],
      cov: ["{\\text{Cov}\\left(#1, #2\\right)}", 2],
      
      // Anomaly detection specific
      anomalyscore: ["{S_{\\text{anomaly}}(#1)}", 1],
      threshold: "{\\tau}",
      isolationpath: ["{h(#1)}", 1],
      lof: ["{\\text{LOF}(#1)}", 1],
      
      // Distance and similarity measures
      euclidean: ["{d_{\\text{euclidean}}(#1, #2)}", 2],
      mahalanobis: ["{d_{\\text{mahalanobis}}(#1, #2)}", 2],
      cosine: ["{\\text{cosine}(#1, #2)}", 2],
      
      // Algorithm-specific notation
      isolationforest: "{\\text{IF}}",
      oneclasssvm: "{\\text{OCSVM}}",
      autoencoder: "{\\text{AE}}",
      
      // Statistical measures
      mean: ["{\\bar{#1}}", 1],
      std: ["{\\sigma_{#1}}", 1],
      
      // Set notation
      dataset: ["{\\mathcal{D}}", 0],
      testset: ["{\\mathcal{T}}", 0],
      anomalies: ["{\\mathcal{A}}", 0],
      normal: ["{\\mathcal{N}}", 0]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  svg: {
    fontCache: 'global',
    displayAlign: 'center',
    displayIndent: '0'
  },
  startup: {
    ready: function () {
      MathJax.startup.defaultReady();
      
      // Custom styling for equations in dark mode
      const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
          if (mutation.type === 'attributes' && mutation.attributeName === 'data-md-color-scheme') {
            const isDark = document.body.getAttribute('data-md-color-scheme') === 'slate';
            document.querySelectorAll('mjx-container').forEach(container => {
              container.style.color = isDark ? '#ffffff' : '#000000';
            });
          }
        });
      });
      
      observer.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-md-color-scheme']
      });
    }
  }
};