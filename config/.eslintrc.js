module.exports = {
    env: {
        browser: true,
        es2021: true,
        node: true
    },
    extends: [
        'eslint:recommended'
    ],
    parserOptions: {
        ecmaVersion: 2021,
        sourceType: 'module'
    },
    rules: {
        // Error prevention
        'no-console': 'warn',
        'no-debugger': 'error',
        'no-unused-vars': 'error',
        'no-undef': 'error',

        // Code style
        'indent': ['error', 2],
        'quotes': ['error', 'single'],
        'semi': ['error', 'always'],
        'comma-dangle': ['error', 'never'],

        // Best practices
        'eqeqeq': 'error',
        'curly': 'error',
        'no-eval': 'error',
        'no-implied-eval': 'error',
        'no-new-func': 'error',
        'no-alert': 'warn',

        // ES6+
        'prefer-const': 'error',
        'no-var': 'error',
        'arrow-spacing': 'error',
        'object-shorthand': 'error'
    },
    globals: {
        // Web APIs
        'fetch': 'readonly',
        'localStorage': 'readonly',
        'sessionStorage': 'readonly',

        // Third-party libraries (based on package.json dependencies)
        'htmx': 'readonly',
        'Alpine': 'readonly',
        'd3': 'readonly',
        'echarts': 'readonly',
        'Sortable': 'readonly',
        'Fuse': 'readonly'
    }
};
