module.exports = {
    extends: [],
    plugins: [],
    rules: {
        // Error prevention
        'no-duplicate-selectors': true,
        'no-empty-source': null,
        'no-invalid-double-slash-comments': true,
        
        // Code style
        'indentation': 2,
        'string-quotes': 'single',
        'color-hex-case': 'lower',
        'color-hex-length': 'short',
        'number-leading-zero': 'always',
        'length-zero-no-unit': true,
        
        // Property formatting
        'property-case': 'lower',
        'declaration-colon-space-after': 'always',
        'declaration-colon-space-before': 'never',
        'declaration-block-semicolon-newline-after': 'always',
        'declaration-block-semicolon-space-before': 'never',
        'declaration-block-trailing-semicolon': 'always',
        
        // Selector formatting
        'selector-type-case': 'lower',
        'selector-combinator-space-after': 'always',
        'selector-combinator-space-before': 'always',
        'selector-list-comma-newline-after': 'always',
        'selector-list-comma-space-before': 'never',
        
        // Media queries
        'media-feature-colon-space-after': 'always',
        'media-feature-colon-space-before': 'never',
        'media-feature-parentheses-space-inside': 'never',
        'media-query-list-comma-newline-after': 'always-multi-line',
        'media-query-list-comma-space-after': 'always-single-line',
        'media-query-list-comma-space-before': 'never',
        
        // At-rules
        'at-rule-name-case': 'lower',
        'at-rule-name-space-after': 'always',
        'at-rule-semicolon-newline-after': 'always',
        
        // Comments
        'comment-whitespace-inside': 'always',
        
        // Block formatting
        'block-closing-brace-newline-after': 'always',
        'block-closing-brace-newline-before': 'always',
        'block-opening-brace-newline-after': 'always',
        'block-opening-brace-space-before': 'always',
        
        // Function formatting
        'function-comma-space-after': 'always',
        'function-comma-space-before': 'never',
        'function-parentheses-space-inside': 'never',
        'function-url-quotes': 'always',
        
        // Value formatting
        'value-list-comma-space-after': 'always',
        'value-list-comma-space-before': 'never'
    },
    ignoreFiles: [
        '**/node_modules/**',
        '**/dist/**',
        '**/build/**'
    ]
};
