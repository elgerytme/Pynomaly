module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testMatch: [
    '**/__tests__/**/*.test.(ts|tsx|js)',
    '**/*.(test|spec).(ts|tsx|js)'
  ],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
    '^.+\\.(js|jsx)$': 'babel-jest'
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.{ts,tsx}',
    '!src/**/*.config.{ts,tsx}',
    '!src/**/index.{ts,tsx}'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html', 'json'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  testTimeout: 10000,
  projects: [
    {
      displayName: 'node',
      testEnvironment: 'node',
      testMatch: ['**/*.node.test.(ts|js)']
    },
    {
      displayName: 'jsdom',
      testEnvironment: 'jsdom',
      testMatch: ['**/*.browser.test.(ts|tsx|js)']
    },
    {
      displayName: 'unit',
      testMatch: ['**/*.unit.test.(ts|tsx|js)']
    },
    {
      displayName: 'integration',
      testMatch: ['**/*.integration.test.(ts|tsx|js)']
    }
  ],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1'
  },
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.test.json'
    }
  }
};