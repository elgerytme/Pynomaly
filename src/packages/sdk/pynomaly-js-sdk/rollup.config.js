import typescript from 'rollup-plugin-typescript2';
import pkg from './package.json';

export default [
  // UMD build
  {
    input: 'src/index.ts',
    output: {
      file: pkg.main,
      format: 'umd',
      name: 'PynomalySDK',
      sourcemap: true,
      globals: {
        'axios': 'axios',
        'react': 'React',
        'react-dom': 'ReactDOM'
      }
    },
    external: ['axios', 'react', 'react-dom'],
    plugins: [
      typescript({
        typescript: require('typescript'),
        tsconfig: './tsconfig.json',
        useTsconfigDeclarationDir: true
      })
    ]
  },
  // ES module build
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'esm',
      sourcemap: true
    },
    external: ['axios', 'react', 'react-dom'],
    plugins: [
      typescript({
        typescript: require('typescript'),
        tsconfig: './tsconfig.json',
        useTsconfigDeclarationDir: true
      })
    ]
  }
];