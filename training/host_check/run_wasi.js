// emg2_host.wasm을 Node의 WASI(샌드박스)로 실행합니다. host_check.py가 부릅니다.
//   node run_wasi.js <emg2_host.wasm> <작업 폴더> <입력 파일 이름> <출력 파일 이름> [채널별 보정값 ...]
// Windows가 새로 빌드한 실행 파일(.exe)을 차단하는 PC에서 쓰는 대체 경로입니다.
// 프로그램은 작업 폴더 안의 파일만 읽고 쓸 수 있습니다.
const fs = require('fs');
const { WASI } = require('node:wasi');

const [wasmPath, workDir, inName, outName, ...gains] = process.argv.slice(2);
const wasi = new WASI({
  version: 'preview1',
  args: ['emg2_host', '/work/' + inName, '/work/' + outName, ...gains],
  preopens: { '/work': workDir },
});
const instance = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)), wasi.getImportObject());
wasi.start(instance);
