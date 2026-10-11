// Optional, dependency-free browser integration test. Node 22+ and Edge/Chrome.
import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {mkdir, readFile, writeFile, rm} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifacts = path.join(root, "browser-validation", `run-${process.pid}-${Date.now()}`);
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
  await run("document.getElementById('check-model').click()");
  await until(() => run("document.getElementById('model-state').textContent.startsWith('Ready at last check')"), "local readiness without description");
  assert.equal(urls.filter(url => url.includes("/api/")).length, 0, "No inference before consent");
  await run(`window.readyFetch = window.fetch; window.fetch = async (...args) => args[0] === "/ready"
    ? new Response(JSON.stringify({status:"unavailable"}), {status:503,headers:{"Content-Type":"application/json"}})
    : window.readyFetch(...args); document.getElementById("check-model").click();`);
  await until(() => run("document.getElementById('model-state').textContent.startsWith('Local model not ready')"), "readiness recovery");
  await run("window.fetch = window.readyFetch");
  await run("document.getElementById('consent').click(); document.querySelector('[data-preset]').click(); document.getElementById('find').click()");
  await until(() => run("document.querySelectorAll('.resource').length > 0"), "ranked supports");
  assert.ok((await run("document.getElementById('results').textContent")).includes("School journey support desk"));
  await run("document.getElementById('read-results').click()");
  assert.equal(await run("document.activeElement.id"), "results-title", "Explicit reading action moves focus without automatic request focus theft");
  assert.equal(await run("document.querySelector('.resource details summary').textContent.startsWith('All ')"), true);
  await run("document.querySelector('.resource details summary').click()");
  assert.equal(await run("document.querySelector('.resource details').open"), true, "Complete explanation is expandable");
  await run("document.getElementById('explain-retrieval').click()");
  await until(() => run("document.getElementById('retrieval-audit').open"), "retrieval limit explanation");
  assert.equal(await run("document.querySelectorAll('#audit-candidates li').length"), 12);
  assert.equal(await run("document.activeElement.id"), "audit-title");
  assert.ok((await run("document.getElementById('audit-candidates').textContent")).includes("not a denial"));
  await run("document.getElementById('close-audit').click()");
  await until(() => run("document.activeElement.id === 'explain-retrieval'"), "audit focus return");
  const lightShot = await rpc("Page.captureScreenshot", {format: "png", captureBeyondViewport: true});
  await writeFile(path.join(artifacts, "desktop-light.png"), Buffer.from(lightShot.data, "base64"));
  await run("document.querySelector('.resource button').click()");
  await until(() => run("document.getElementById('handoff').open"), "handoff simulation");
  assert.ok((await run("document.getElementById('handoff-message').textContent")).includes("Nothing was sent"));
  assert.equal(await run("document.activeElement.id"), "handoff-title", "Modal summary receives initial focus");
  await rpc("Input.dispatchKeyEvent", {type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9});
  await rpc("Input.dispatchKeyEvent", {type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9});
  assert.equal(await run("document.activeElement.id"), "close-dialog");
  await rpc("Input.dispatchKeyEvent", {type: "keyDown", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27});
  await rpc("Input.dispatchKeyEvent", {type: "keyUp", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27});
  await until(() => run("!document.getElementById('handoff').open"), "keyboard modal dismissal");
  await until(() => run("document.activeElement.getAttribute('aria-label')?.startsWith('Preview human handoff')"), "handoff focus restored");
  assert.equal(await run("document.activeElement.getAttribute('aria-label')?.startsWith('Preview human handoff')"), true);
  await run("document.getElementById('consent').click()");
  assert.equal(await run("document.getElementById('barrier').value === '' && !document.querySelector('.resource') && !document.getElementById('handoff').open"), true);
  assert.equal(await run("document.getElementById('explain-retrieval').disabled && document.getElementById('audit-candidates').children.length === 0"), true, "Withdrawal clears retrieval audit");
  assert.equal(await run("document.getElementById('read-results').disabled"), true);
  await run("document.getElementById('consent').click(); document.getElementById('barrier').value = 'quantum entanglement'; document.getElementById('barrier').dispatchEvent(new Event('input')); document.getElementById('find').click()");
  await until(() => run("document.getElementById('result-count').textContent === '0 demo ideas'"), "no-match state");
  assert.ok((await run("document.getElementById('results').textContent")).includes("not a denial"));
  assert.ok((await run("document.getElementById('results').textContent")).includes("None of the usable words"));
  await run("document.getElementById('barrier').value = 'printed worksheets paper exercises'; document.getElementById('barrier').dispatchEvent(new Event('input')); document.getElementById('mode').value = 'online'; document.getElementById('mode').dispatchEvent(new Event('change')); document.getElementById('find').click()");
  await until(() => run("!!document.getElementById('relax-format')"), "format diagnostic recovery");
  await run("document.getElementById('relax-format').click()");
  await until(() => run("document.querySelectorAll('.resource').length > 0"), "explicit any-format retry");
  assert.equal(await run("document.getElementById('mode').value"), "any");
  await run("document.getElementById('barrier').value = 'without internet'; document.getElementById('barrier').dispatchEvent(new Event('input')); document.getElementById('find').click()");
  await until(() => run("document.getElementById('status').textContent.includes('does not understand negation')"), "word model limitation warning");
  await run("document.getElementById('result-limit').value = '1'; document.getElementById('result-limit').dispatchEvent(new Event('change')); document.getElementById('find').click()");
  await until(() => run("document.querySelectorAll('.resource').length === 1"), "one-at-a-time review");
  const beforeRejected = urls.filter(url => url.includes("/api/match")).length;
  await run("document.getElementById('barrier').value = 'a@example.invalid'; document.getElementById('barrier').dispatchEvent(new Event('input')); document.getElementById('find').click()");
  await until(() => run("document.getElementById('status').dataset.error === 'true'"), "invalid input");
  assert.ok((await run("document.getElementById('status').textContent")).includes("Remove"));
  assert.equal(urls.filter(url => url.includes("/api/match")).length, beforeRejected, "Obvious identifiers rejected before transport");
  assert.equal(await run("document.getElementById('barrier').getAttribute('aria-invalid')"), "true");
  assert.equal(await run("document.activeElement.id"), "barrier");
  await run("document.getElementById('barrier').value = 'bus\\u202efares'; document.getElementById('barrier').dispatchEvent(new Event('input')); document.getElementById('find').click()");
  assert.ok((await run("document.getElementById('barrier-error').textContent")).includes("directional"));
  assert.equal(urls.filter(url => url.includes("/api/match")).length, beforeRejected);
  await run("document.getElementById('clear').click()");
  assert.equal(await run("!document.getElementById('consent').checked && document.getElementById('find').disabled"), true);
  assert.equal(await run("document.getElementById('barrier').getAttribute('aria-invalid')"), "false");

  await run(`window.savedFetch = window.fetch; window.fetch = async (...args) => {
    const response = await window.savedFetch(...args);
    await new Promise(resolve => setTimeout(resolve, 350)); return response;
  }; document.getElementById('consent').click(); document.querySelector('[data-preset]').click();
  document.getElementById('find').click(); document.getElementById('cancel-request').click();`);
  await delay(600);
  assert.equal(await run("document.getElementById('consent').checked && document.getElementById('barrier').value !== '' && !document.querySelector('.resource') && document.getElementById('cancel-request').hidden"), true, "Cancel preserves example but gates late response");
  await run("window.fetch = window.savedFetch; document.getElementById('clear').click()");
  assert.equal(await run("document.getElementById('browse').disabled"), true);
  await run("document.getElementById('consent').click(); document.getElementById('browse').click()");
  await until(() => run("document.querySelectorAll('.resource').length === 6"), "catalog browse without input");
  assert.equal(await run("document.getElementById('barrier').value"), "");
  assert.equal(await run("!!document.querySelector('.score')"), false, "Catalog is not ranked");
  await run("document.querySelectorAll('[data-choice]')[0].click(); document.querySelectorAll('[data-choice]')[1].click(); document.querySelector('#choice-list button').focus(); document.querySelector('#choice-list button').click()");
  assert.equal(await run("document.activeElement.closest('#choice-list') !== null"), true, "Removing focused choice advances to remaining choice");
  await run("document.querySelectorAll('[data-choice]')[0].click(); document.getElementById('compare').click()");
  await until(() => run("document.getElementById('comparison').open"), "comparison dialog");
  assert.equal(await run("document.getElementById('comparison-cards').children.length"), 2);
  assert.ok((await run("document.getElementById('comparison-message').textContent")).includes("No option is preferred"));
  await run("document.getElementById('close-comparison').click()");
  await until(() => run("document.activeElement.id === 'compare'"), "comparison focus restored");
  await run("document.getElementById('prepare-plan').click()");
  await until(() => run("document.getElementById('planning').open"), "fictional planning checklist");
  assert.equal(await run("document.getElementById('print-plan').disabled"), true);
  assert.equal(await run("document.getElementById('plan-resources').children.length"), 2);
  await run("document.querySelector('#plan-verification input').click()");
  assert.ok((await run("document.getElementById('review-progress').textContent")).startsWith("1 of 4"));
  await run("document.getElementById('reset-review').click()");
  assert.equal(await run("document.querySelectorAll('#plan-verification input:checked').length"), 0);
  assert.equal(await run("document.activeElement.id"), "verify-0");
  await run("document.querySelector('#plan-verification input').click()");
  await run("window.printed = false; window.print = () => { window.printed = document.body.dataset.printPlan === 'true'; }; document.getElementById('print-consent').click(); document.getElementById('print-plan').click()");
  assert.equal(await run("window.printed && !document.body.dataset.printPlan"), true, "Print requires explicit transient authorization");
  await rpc("Emulation.setEmulatedMedia", {media: "print"});
  assert.equal(await run("getComputedStyle(document.querySelector('main')).display"), "none", "Print never exposes barrier form");
  assert.equal(await run("getComputedStyle(document.getElementById('planning')).display"), "none", "Direct printing has no authorized plan");
  await rpc("Emulation.setEmulatedMedia", {media: ""});
  await run("document.getElementById('close-plan').click()");
  await until(() => run("document.activeElement.id === 'prepare-plan'"), "plan focus restored");
  assert.equal(await run("document.getElementById('print-consent').checked"), false);
  assert.equal(await run("document.querySelectorAll('#plan-verification input:checked').length"), 0, "Closing the preview discards review marks, not verified outcomes");
  await run("document.getElementById('clear-choices').click()");
  assert.equal(await run("document.getElementById('choice-list').children.length === 0 && document.getElementById('consent').checked && document.querySelectorAll('.resource').length === 6"), true, "Choice-only reset preserves consent and cards");
  assert.equal(await run("document.getElementById('plan-resources').children.length"), 0);
  await run("document.getElementById('next-page').click()");
  await until(() => run("document.getElementById('result-count').textContent === '7–12 of 12 catalog ideas'"), "catalog pagination");
  await run("document.getElementById('previous-page').click()");
  await until(() => run("document.getElementById('result-count').textContent === '1–6 of 12 catalog ideas'"), "catalog backward navigation");
  assert.equal(await run("document.getElementById('previous-page').disabled"), true);
  await run("document.getElementById('catalog-sort').value = 'title'; document.getElementById('catalog-sort').dispatchEvent(new Event('change')); document.getElementById('browse').click()");
  await until(() => run("document.querySelectorAll('.resource').length === 6"), "alphabetic catalog order");
  const titles = await run("[...document.querySelectorAll('.resource h3')].map(el => el.textContent)");
  assert.deepEqual(titles, [...titles].sort());
  await run("document.getElementById('category').value = 'Food'; document.getElementById('category').dispatchEvent(new Event('change')); document.getElementById('browse').click()");
  await until(() => run("document.querySelectorAll('.resource').length === 1"), "catalog category filter");
  assert.ok((await run("document.getElementById('results').textContent")).includes("Food and learning"));
  assert.ok((await run("document.getElementById('catalog-coverage').textContent")).includes("1 offline, 0 online"));
  assert.ok((await run("document.querySelector('#category option[value=\"Food\"]').textContent")).includes("1 in this format"));
  await run("document.getElementById('clear').click()");
  assert.equal(await run("document.getElementById('choice-list').children.length"), 0, "Reset clears temporary choices");
  await run("document.getElementById('consent').click(); document.getElementById('browse').click()");
  await until(() => run("document.querySelectorAll('[data-related]').length === 6"), "related browsing controls");
  await run("document.querySelector('[data-related]').click()");
  await until(() => run("document.getElementById('result-count').textContent.includes('related catalog ideas')"), "related catalog response");
  assert.equal(await run("document.getElementById('barrier').value"), "", "Related descriptions never need a barrier");
  await run("document.getElementById('clear').click()");

  // Artificially delay the response to ensure revocation cannot render late results.
  await run(`window.escapeFetch = window.fetch; window.fetch = async (...args) => {
    const response = await window.escapeFetch(...args);
    await new Promise(resolve => setTimeout(resolve, 350)); return response;
  }; document.getElementById('consent').click(); document.querySelector('[data-preset]').click(); document.getElementById('find').click();`);
  await rpc("Input.dispatchKeyEvent", {type:"keyDown",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
  await rpc("Input.dispatchKeyEvent", {type:"keyUp",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
  await delay(600);
  assert.equal(await run("document.getElementById('consent').checked && !document.querySelector('.resource') && document.getElementById('barrier').value !== '' && document.activeElement.id === 'barrier'"), true, "Escape cancels pending work and gates late output");
  await run("window.fetch = window.escapeFetch; document.getElementById('clear').click()");
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
  assert.equal(await run("document.getElementById('theme').getAttribute('aria-label')"), "Use light theme; currently dark");
  assert.equal(await run("document.documentElement.scrollWidth <= window.innerWidth"), true, "Mobile has no horizontal overflow");
  assert.equal(await run("localStorage.length === 0 && sessionStorage.length === 0"), true);
  const darkShot = await rpc("Page.captureScreenshot", {format: "png", captureBeyondViewport: true});
  await writeFile(path.join(artifacts, "mobile-dark.png"), Buffer.from(darkShot.data, "base64"));
  await rpc("Page.navigate", {url:base});
  await waitPage();
  await rpc("Emulation.setEmulatedMedia", {features:[{name:"prefers-color-scheme",value:"dark"}]});
  await until(() => run("document.documentElement.dataset.theme === 'dark'"), "system dark preference");
  await rpc("Emulation.setEmulatedMedia", {features:[{name:"prefers-color-scheme",value:"light"}]});
  await until(() => run("document.documentElement.dataset.theme === 'light'"), "live system light preference");
  await run("document.getElementById('theme').click()");
  await rpc("Emulation.setEmulatedMedia", {features:[{name:"prefers-color-scheme",value:"light"}]});
  assert.equal(await run("document.documentElement.dataset.theme"), "dark", "Manual choice overrides system changes for this page");
  const beforeNoScript = urls.filter(url => url.includes("/api/")).length;
  await rpc("Emulation.setScriptExecutionDisabled", {value:true});
  await rpc("Page.navigate", {url:base + "/?clawpilotTheme=light"});
  await waitPage();
  assert.ok((await run("document.getElementById('no-script').textContent")).includes("JavaScript is disabled"));
  assert.equal(await run("getComputedStyle(document.getElementById('no-script')).display !== 'none' && document.getElementById('find').disabled && document.getElementById('barrier').disabled"), true, "No-script fallback is visible and consent controls stay inert");
  assert.equal(urls.filter(url => url.includes("/api/")).length, beforeNoScript, "No inference with scripts disabled");
  await rpc("Emulation.setScriptExecutionDisabled", {value:false});
  assert.ok(urls.every(url => url.startsWith(base)), `Page sends only loopback requests: ${JSON.stringify(urls.filter(url => !url.startsWith(base)).map(url => url.slice(0, 160)))}`);
  assert.deepEqual(errors, [], "No browser exceptions or server stderr");
  console.log("PASS: consent, real HTTP ranking, handoff, withdrawal, no-match, invalid input, reset, late-response gate, themes, mobile layout, no storage, no external requests.");
} catch (error) {
  console.error(error.stack || error.message);
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
