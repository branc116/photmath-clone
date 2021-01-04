import ts from "@rollup/plugin-typescript";
export default {
    input: 'index.ts',
    external: ["@tensorflow/tfjs"],
    output: {
        file: 'index.js',
        compact: true,
        format: 'iife',
        globals: {
            "@tensorflow/tfjs": "tf"
        }
    },
    plugins: [ts({
        "target": "ESNext",                          /* Specify ECMAScript target version: 'ES3' (default), 'ES5', 'ES2015', 'ES2016', 'ES2017', 'ES2018', 'ES2019', 'ES2020', or 'ESNEXT'. */
        "module": "ESNext",                     /* Specify module code generation: 'none', 'commonjs', 'amd', 'system', 'umd', 'es2015', 'es2020', or 'ESNext'. */
        "lib": ["DOM", "ES2020"],                             /* Specify library files to be included in the compilation. */
        "strict": true,                           /* Enable all strict type-checking options. */
        "moduleResolution": "node",            /* Specify module resolution strategy: 'node' (Node.js) or 'classic' (TypeScript pre-1.6). */
        "esModuleInterop": true,                  /* Enables emit interoperability between CommonJS and ES Modules via creation of namespace objects for all imports. Implies 'allowSyntheticDefaultImports'. */
        "skipLibCheck": true,                     /* Skip type checking of declaration files. */
        "forceConsistentCasingInFileNames": true  /* Disallow inconsistently-cased references to the same file. */
    })]
};
