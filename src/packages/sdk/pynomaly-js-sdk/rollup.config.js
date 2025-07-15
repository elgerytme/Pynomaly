import typescript from 'rollup-plugin-typescript2';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';
import { readFileSync } from 'fs';

const isProduction = process.env.NODE_ENV === 'production';
const target = process.env.TARGET || 'universal';

const commonConfig = {
  input: 'src/index.ts',
  external: ['axios', 'eventemitter3'],
  plugins: [
    resolve({
      browser: target === 'browser',
      preferBuiltins: target === 'node'
    }),
    commonjs(),
    typescript({
      tsconfig: './tsconfig.json',
      clean: true,
      exclude: ['**/*.test.ts', '**/*.spec.ts', 'tests/**/*']
    }),
    ...(isProduction ? [terser()] : [])
  ]
};

const configs = [];

// Universal build (default)
if (target === 'universal' || !target) {
  configs.push({
    ...commonConfig,
    output: [
      {
        file: 'dist/index.js',
        format: 'cjs',
        exports: 'named'
      },
      {
        file: 'dist/index.esm.js',
        format: 'es'
      }
    ]
  });
}

// Node.js specific build
if (target === 'node') {
  configs.push({
    ...commonConfig,
    external: [...commonConfig.external, 'ws', 'node-fetch'],
    output: {
      file: 'dist/index.node.js',
      format: 'cjs',
      exports: 'named'
    }
  });
}

// Browser specific build
if (target === 'browser') {
  configs.push({
    ...commonConfig,
    output: {
      file: 'dist/index.browser.js',
      format: 'umd',
      name: 'Pynomaly',
      globals: {
        'axios': 'axios',
        'eventemitter3': 'EventEmitter3'
      }
    }
  });
}

export default configs;