// emg2_host.wasm을 Node의 WASI(샌드박스)로 실행합니다. host_check.py가 부릅니다.
//   node run_wasi.js <emg2_host.wasm> <작업 폴더> <emg2_host에 넘길 인자 ...>
// 인자 중 파일은 '/work/파일이름'으로 넘깁니다 (/work = 작업 폴더).
// Windows가 새로 빌드한 실행 파일(.exe)을 차단하는 PC에서 쓰는 대체 경로입니다.
// 프로그램은 작업 폴더 안의 파일만 읽고 쓸 수 있습니다.
const fs = require('fs');
const { WASI } = require('node:wasi');

const [wasmPath, workDir, ...rest] = process.argv.slice(2);
const wasi = new WASI({
  version: 'preview1',
  args: ['emg2_host', ...rest],
  preopens: { '/work': workDir },
});
const instance = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)), wasi.getImportObject());
wasi.start(instance);
