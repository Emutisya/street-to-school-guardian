// Optional, dependency-free browser integration test. Node 22+ and Edge/Chrome.
import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {mkdir, readFile, writeFile, rm} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifacts = path.join(root, "browser-validation");
const profile = path.join(artifacts, "edge-profile");
const browserPath = process.env.GUARDIAN_BROWSER;
if (!browserPath) throw new Error("Set GUARDIAN_BROWSER to an installed Edge or Chrome executable.");
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, label) {
  for (let n = 0; n < 150; n++) {
    const result = await check();
    if (result) return result;
    await delay(100);
  }
  throw new Error(`Timed out: ${label}`);
}

let server, browser, socket, rpc;
const errors = [], urls = [];
let output = "", browserError = "";
try {
  await mkdir(artifacts, {recursive: true});
  await rm(profile, {recursive: true, force: true});
  server = spawn(process.env.PYTHON || "python", ["-m", "guardian", "serve", "--port", "0"], {cwd: root});
  server.stdout.on("data", chunk => output += chunk.toString());
  server.stderr.on("data", chunk => errors.push(chunk.toString()));
  const base = await until(() => /http:\/\/127\.0\.0\.1:\d+/.exec(output)?.[0], "server startup");
  assert.equal((await (await fetch(base + "/health")).json()).status, "ok");
  browser = spawn(browserPath, [
    "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    "--disable-background-networking", "--disable-extensions",
    "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"
  ], {cwd: root, stdio: ["ignore", "ignore", "pipe"]});
  browser.stderr.on("data", chunk => browserError += chunk.toString());
  const port = await until(async () => {
    try { return (await readFile(path.join(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]; }
    catch { return null; }
  }, "browser debugging endpoint");
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const target = targets.find(item => item.type === "page");
  assert.ok(target, "Browser page target");
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, {once: true});
    socket.addEventListener("error", reject, {once: true});
  });
  let id = 0;
  const waiting = new Map();
  socket.addEventListener("message", event => {
    const data = JSON.parse(event.data);
    if (data.id && waiting.has(data.id)) {
      const {resolve, reject} = waiting.get(data.id);
      waiting.delete(data.id);
      data.error ? reject(new Error(JSON.stringify(data.error))) : resolve(data.result);
    }
    if (data.method === "Runtime.exceptionThrown") errors.push(JSON.stringify(data.params.exceptionDetails));
    if (data.method === "Network.requestWillBeSent") urls.push(data.params.request.url);
  });
  rpc = (method, params = {}) => new Promise((resolve, reject) => {
    const requestId = ++id;
    const timeout = setTimeout(() => { waiting.delete(requestId); reject(new Error(`CDP timeout: ${method}`)); }, 10000);
    waiting.set(requestId, {
      resolve: result => { clearTimeout(timeout); resolve(result); },
      reject: error => { clearTimeout(timeout); reject(error); }
    });
    socket.send(JSON.stringify({id: requestId, method, params}));
  });
  const run = async expression => {
    const result = await rpc("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitPage = () => until(() => run("document.readyState === 'complete' && !!document.getElementById('consent')"), "page ready");
  await rpc("Runtime.enable");
  await rpc("Network.enable");
  await rpc("Page.enable");
  await rpc("Emulation.setDeviceMetricsOverride", {width: 1360, height: 1000, deviceScaleFactor: 1, mobile: false});
  await rpc("Page.navigate", {url: base + "/?clawpilotTheme=light"});
  await waitPage();
  assert.equal(await run("document.documentElement.dataset.theme"), "light");
  assert.equal(await run("document.getElementById('find').disabled && document.getElementById('barrier').disabled"), true);
  assert.equal(urls.filter(url => url.includes("/api/")).length, 0, "No inference before consent");
  await run("document.getElementById('consent').click(); document.querySelector('[data-preset]').click(); document.getElementById('find').click()");
  await until(() => run("document.querySelectorAll('.resource').length > 0"), "ranked supports");
  assert.ok((await run("document.getElementById('results').textContent")).includes("School journey support desk"));
  const lightShot = await rpc("Page.captureScreenshot", {format: "png", captureBeyondViewport: true});
  await writeFile(path.join(artifacts, "desktop-light.png"), Buffer.from(lightShot.data, "base64"));
  await run("document.querySelector('.resource button').click()");
  await until(() => run("document.getElementById('handoff').open"), "handoff simulation");
  assert.ok((await run("document.getElementById('handoff-message').textContent")).includes("Nothing was sent"));
  await run("document.getElementById('close-dialog').click(); document.getElementById('consent').click()");
  assert.equal(await run("document.getElementById('barrier').value === '' && !document.querySelector('.resource') && !document.getElementById('handoff').open"), true);
  await run("document.getElementById('consent').click(); document.getElementById('barrier').value = 'quantum entanglement'; document.getElementById('barrier').dispatchEvent(new Event('input')); document.getElementById('find').click()");
  await until(() => run("document.getElementById('result-count').textContent === '0 demo ideas'"), "no-match state");
  assert.ok((await run("document.getElementById('results').textContent")).includes("not a denial"));
  await run("document.getElementById('barrier').value = 'a@example.invalid'; document.getElementById('barrier').dispatchEvent(new Event('input')); document.getElementById('find').click()");
  await until(() => run("document.getElementById('status').dataset.error === 'true'"), "invalid input");
  assert.ok((await run("document.getElementById('status').textContent")).includes("Remove"));
  await run("document.getElementById('clear').click()");
  assert.equal(await run("!document.getElementById('consent').checked && document.getElementById('find').disabled"), true);

  // Artificially delay the response to ensure revocation cannot render late results.
  await run(`window.originalFetch = window.fetch; window.fetch = async (...args) => {
    const response = await window.originalFetch(...args);
    await new Promise(resolve => setTimeout(resolve, 350));
    return response;
  }; document.getElementById('consent').click(); document.querySelector('[data-preset]').click(); document.getElementById('find').click(); document.getElementById('consent').click();`);
  await delay(600);
  assert.equal(await run("!document.querySelector('.resource') && document.getElementById('barrier').value === ''"), true);
  await run("window.fetch = window.originalFetch");
  await rpc("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await run("document.getElementById('theme').click()");
  await delay(200);
  assert.equal(await run("document.documentElement.dataset.theme"), "dark");
  assert.equal(await run("document.documentElement.scrollWidth <= window.innerWidth"), true, "Mobile has no horizontal overflow");
  assert.equal(await run("localStorage.length === 0 && sessionStorage.length === 0"), true);
  const darkShot = await rpc("Page.captureScreenshot", {format: "png", captureBeyondViewport: true});
  await writeFile(path.join(artifacts, "mobile-dark.png"), Buffer.from(darkShot.data, "base64"));
  assert.ok(urls.every(url => url.startsWith(base)), "Page sends only loopback requests");
  assert.deepEqual(errors, [], "No browser exceptions or server stderr");
  console.log("PASS: consent, real HTTP ranking, handoff, withdrawal, no-match, invalid input, reset, late-response gate, themes, mobile layout, no storage, no external requests.");
} catch (error) {
  console.error(error.message);
  if (browserError && !socket) console.error(browserError.slice(-1500));
  process.exitCode = 1;
} finally {
  if (rpc && socket?.readyState === WebSocket.OPEN) {
    try { await rpc("Browser.close"); } catch {}
    socket.close();
  }
  if (browser?.exitCode === null) { await delay(500); if (browser.exitCode === null) browser.kill(); }
  if (server?.exitCode === null) server.kill();
  await delay(500);
  await rm(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 200});
}
