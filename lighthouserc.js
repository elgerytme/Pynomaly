module.exports = {
  ci: {
    collect: {
      url: [
        'http://localhost:8000',
        'http://localhost:8000/dashboard',
        'http://localhost:8000/detection',
        'http://localhost:8000/datasets'
      ],
      numberOfRuns: 3,
      settings: {
        chromeFlags: '--no-sandbox --headless --disable-gpu',
        preset: 'desktop',
      },
    },
    assert: {
      assertions: {
        'categories:performance': ['warn', { minScore: 0.8 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['warn', { minScore: 0.8 }],
        'categories:seo': ['warn', { minScore: 0.8 }],
        'categories:pwa': ['warn', { minScore: 0.7 }],
      },
    },
    upload: {
      target: 'filesystem',
      outputDir: './test_reports/lighthouse',
    },
  },
};
