// Compile a .ksy to Python with the official Kaitai Struct compiler (JS build).
// usage: node compile.js input.ksy outdir
const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");
const KaitaiStructCompiler = require("kaitai-struct-compiler");
const [, , input, outdir] = process.argv;
const spec = yaml.load(fs.readFileSync(input, "utf8"));
const compiler = new KaitaiStructCompiler();
compiler.compile("python", spec, null, false).then((files) => {
  for (const [name, src] of Object.entries(files)) {
    fs.writeFileSync(path.join(outdir, name), src);
    console.log(path.join(outdir, name));
  }
}).catch((e) => { console.error("KSC ERROR:", e && (e.message || e.toString())); process.exit(2); });
