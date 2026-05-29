// Records a ProofMetaTutor demo: a video walkthrough + key screenshots.
// Usage:
//   DEMO_URL=<your Cloud Run URL> node e2e/record-demo.mjs
//   (defaults to http://localhost:3100)
// Output: apps/web/demo-media/  (prooftutor-demo.webm + NN-*.png)
import { chromium } from "playwright";
import { mkdir, readdir, rename } from "node:fs/promises";
import path from "node:path";

const BASE = process.env.DEMO_URL || "http://localhost:3100";
const OUT = path.resolve("demo-media");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function clickChip(page, text) {
  try {
    await page.locator(".chip", { hasText: text }).first().click({ timeout: 8000 });
    await sleep(450);
  } catch {
    /* label mismatch or not found — skip, keep recording */
  }
}

async function runVerify(page, shot) {
  await page.getByRole("button", { name: "검증 실행" }).click();
  try {
    await page.waitForSelector(".verdict", { timeout: 20000 });
    await sleep(1400); // count-up + bar animation
  } catch {
    await sleep(2000); // screenshot whatever is on screen for diagnosis
  }
  await page.screenshot({ path: path.join(OUT, shot) });
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1366, height: 900 },
    deviceScaleFactor: 2,
    recordVideo: { dir: OUT, size: { width: 1366, height: 900 } },
  });
  const page = await ctx.newPage();
  page.on("console", (m) => console.log(`[console.${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  page.on("requestfailed", (r) => console.log(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`));

  await page.goto(BASE, { waitUntil: "networkidle" });
  await sleep(1200);
  await page.screenshot({ path: path.join(OUT, "01-landing.png") });

  await clickChip(page, "일차방정식");
  await page.screenshot({ path: path.join(OUT, "02-sample-loaded.png") });
  await runVerify(page, "03-verdict-correct.png");

  await clickChip(page, "분수 계산");
  await runVerify(page, "04-verdict-fraction.png");

  await page.locator(".field").nth(0).locator("input").fill("5x = 20");
  await page.locator(".field").nth(1).locator("textarea").fill("양변을 5로 나누면 x = 4");
  await runVerify(page, "05-custom.png");

  // Multi-step solution check — pinpoints the wrong step (the wow moment).
  const checkBtn = page.getByRole("button", { name: "풀이 검사" });
  await checkBtn.scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(OUT, "06-solution-input.png") });
  await checkBtn.click();
  try {
    await page.waitForSelector(".sc-step", { timeout: 30000 });
    await sleep(1500);
  } catch {
    await sleep(2000);
  }
  // Center the flagged step so the verdict + reason are fully visible.
  const firstError = page.locator(".sc-step.first-error");
  if (await firstError.count()) {
    await firstError.scrollIntoViewIfNeeded();
    await sleep(300);
  } else {
    await page.locator(".sc-result").scrollIntoViewIfNeeded();
  }
  await page.screenshot({ path: path.join(OUT, "07-solution-result.png") });

  await ctx.close();
  await browser.close();

  const files = await readdir(OUT);
  const webm = files.find((f) => f.endsWith(".webm") && f !== "prooftutor-demo.webm");
  if (webm) await rename(path.join(OUT, webm), path.join(OUT, "prooftutor-demo.webm"));
  console.log("demo media written to", OUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
