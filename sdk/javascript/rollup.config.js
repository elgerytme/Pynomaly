import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const pkg = require('./package.json');

export default [
  // UMD build
  {
    input: 'src/index.ts',
    output: {
      name: 'PynomalyClient',
      file: pkg.main,
      format: 'umd',
      globals: {
        'axios': 'axios',
        'ws': 'WebSocket'
      }
    },
    external: ['axios', 'ws'],
    plugins: [
      resolve({
        browser: true,
        preferBuiltins: false
      }),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        declaration: true,
        declarationDir: './dist'
      })
    ]
  },
  // ES module build
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'es'
    },
    external: ['axios', 'ws'],
    plugins: [
      resolve({
        browser: true,
        preferBuiltins: false
      }),
      commonjs(),
      typescript({
        tsconfig: './tsconfig.json',
        declaration: false
      })
    ]
  }
];
