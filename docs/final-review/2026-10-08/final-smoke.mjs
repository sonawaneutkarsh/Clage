import { chromium, expect } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const output = path.dirname(fileURLToPath(import.meta.url));
const baseURL = process.argv[2] || 'http://127.0.0.1:8880';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [], downloads = [], screenshots = [];
page.on('pageerror', error => errors.push(error.message));

async function capture(name) {
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(output, name), fullPage: true });
  screenshots.push(name);
}

async function exportFrom(selector) {
  const pending = page.waitForEvent('download');
  await page.locator(selector).click();
  const download = await pending, stream = await download.createReadStream(), chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const bytes = Buffer.concat(chunks);
  expect(bytes.length).toBeGreaterThan(0);
  downloads.push({ filename: download.suggestedFilename(), bytes: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex') });
}

try {
  const created = await page.request.post(`${baseURL}/api/runs`, { data: {
    population: 24, generations: 3, ticks: 20, width: 24, height: 20, food: 80, seed: 42
  } });
  expect(created.ok()).toBe(true);
  await page.goto(baseURL);
  await expect(page.locator('#render-rate')).not.toContainText('—');
  await page.locator('#speed').fill('30');
  await page.locator('#speed').dispatchEvent('change');
  await page.locator('#play').click();
  await expect(page.locator('#tick')).not.toContainText('tick 0 /');
  await page.locator('#play').click();
  await expect(page.locator('#mode')).toContainText('PAUSED');
  await page.locator('#body-choice').selectOption('0');
  await page.locator('#follow').click();
  const observed = await (await page.request.get(`${baseURL}/api/state`)).json();
  const child = observed.frame.organisms.find(body => body.parent === 0);
  expect(child).toBeTruthy();
  await page.locator('[data-view=evolution]').click();
  await page.locator('#lineage .lineage-node').filter({ hasText: new RegExp(`\\b${child.id}\\b`) }).click();
  await expect(page.locator('#view-back')).toHaveText('← Back to organism 0');
  await capture('smoke-body-lineage.png');
  await page.locator('#view-back').click();
  await expect(page.locator('#body-choice')).toHaveValue('0');
  await page.locator('[data-view=ecosystem]').click();
  await capture('smoke-live.png');
  await page.locator('#open-neural').click();
  await page.locator('#network .edge').first().focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#network-detail')).toContainText('"kind": "Connection"');
  await capture('smoke-connection.png');
  await page.locator('#view-back').click();
  await page.locator('#genome-compare').selectOption({ index: 2 });
  await expect(page.locator('#network-compare-card')).toBeVisible();
  await capture('smoke-genome-comparison.png');
  await page.locator('#view-back').click();
  await expect(page.locator('#ecosystem-view')).toBeVisible();
  await expect(page.locator('#follow')).toHaveAttribute('aria-pressed', 'true');
  await page.locator('#play').click();
  await expect(page.locator('#mode')).toContainText('COMPLETE', { timeout: 15000 });
  await page.locator('[data-view=evolution]').click();
  await expect(page.locator('#history tbody tr')).toHaveCount(3);
  await capture('smoke-evolution.png');
  await page.locator('#history button').filter({ hasText: 'Inspect' }).first().click();
  await expect(page.locator('#view-back')).toHaveText('← Back to evolution');
  await page.locator('#view-back').click();
  await exportFrom('#champion-export');
  await page.locator('[data-view=research]').click();
  await page.locator('#evaluate').click();
  await expect(page.locator('#evaluation tbody tr')).toHaveCount(20);
  await exportFrom('#evaluation-export');
  await capture('smoke-research.png');
  await page.locator('[data-view=ecosystem]').click();
  await page.locator('#review').click();
  const replay = await (await page.request.get(`${baseURL}/api/replay`)).body();
  await page.locator('#compare-import').setInputFiles({ name: 'comparison.json', mimeType: 'application/json', buffer: replay });
  await page.locator('#scrub').fill('10');
  await page.locator('#scrub').dispatchEvent('input');
  await expect(page.locator('#replay-position')).toContainText('Frame 11/');
  await capture('smoke-replay-comparison.png');
  for (const selector of ['#csv', '#chart-export', '#screenshot', '#export']) await exportFrom(selector);
  await page.setViewportSize({ width: 390, height: 844 });
  await capture('smoke-mobile-replay.png');
  await page.locator('#live').click();
  await expect(page.locator('.compare-key')).toBeHidden();
  await page.locator('[data-view=neural]').click();
  await page.locator('#network').hover();
  await page.mouse.wheel(300, 0);
  await expect.poll(() => page.locator('#network').evaluate(node => node.parentElement.scrollLeft)).toBeGreaterThan(0);
  await capture('smoke-mobile-neural-scrolled.png');
  await page.goBack();
  await expect(page.locator('#ecosystem-view')).toBeVisible();
  page.once('dialog', dialog => dialog.accept());
  await page.locator('#reset').click();
  await expect(page.locator('#tick')).toContainText('tick 0 /');
  await page.locator('#step').click();
  await expect(page.locator('#tick')).toContainText('tick 1 /');
  expect(errors).toEqual([]);
  await fs.writeFile(path.join(output, 'final-browser-smoke.json'), JSON.stringify({
    status: 'passed', browser: browser.version(), baseURL, errors, downloads, screenshots,
    method: 'Actual headless Chromium UI interactions, three 20-tick generations; separate full-default deterministic validation is recorded elsewhere.'
  }, null, 2));
  console.log(`Final browser smoke passed: ${downloads.length} nonempty exports, ${screenshots.length} screenshots, zero page errors.`);
} finally {
  await browser.close();
}
