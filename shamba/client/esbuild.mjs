import civetPlugin from "@danielx/civet/esbuild";
import * as esbuild from "esbuild";
import envFilePlugin from "esbuild-envfile-plugin";

const config = {
  entryPoints: ["./src"],
  jsxFactory: "h",
  jsxFragment: "Fragment",
  jsx: "automatic",
  bundle: true,
  sourcemap: process.env.NODE_ENV !== "production",
  minify: process.env.NODE_ENV === "production",
  metafile: process.env.NODE_ENV !== "production",
  chunkNames: "[name]",
  target: "es6",
  loader: { ".js": "jsx" },
  alias: { "~": "./src", react: "preact/compat", "react-dom": "preact/compat" },
  outdir: "../server/static/js",
  plugins: [
    envFilePlugin,
    civetPlugin({
      // Options and their defaults:
      // dts: false,                     // generate .d.ts files?
      // outputExtension: '.civet.tsx',  // replaces .civet in output
      js: true, // use Civet's TS -> JS transpiler?
    }),
  ],
};

if (process.argv.includes("--watch")) {
  esbuild
    .context({ ...config, sourcemap: "both" })
    .then((ctx) => {
      ctx.watch();
      console.log("Watching...");
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
} else {
  esbuild
    .build(config)
    .then((result) => {
      console.log("⚡ Done");

      if (process.env.NODE_ENG !== "production") {
        esbuild;
      }
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}
