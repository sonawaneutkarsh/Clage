import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

const config = { population: 12, generations: 2, ticks: 12, width: 16, height: 12, food: 30, regrowth: 1, seed: 42 };

test.beforeEach(async ({ request, page }) => {
  await request.post('/api/runs', { data: config });
  await page.goto('/');
  await expect(page.locator('#population')).not.toHaveText('—');
  await expect(page.locator('#render-rate')).not.toContainText('—');
});

async function pickOrganism(page, request) {
  const data = await (await request.get('/api/state')).json();
  const body = data.frame.organisms.find(organism => organism.alive);
  const bounds = await page.locator('#world').boundingBox();
  const cell = Math.min((bounds.width - 75) / data.config.width, (bounds.height - 65) / data.config.height);
  const ox = bounds.width / 2 - data.config.width / 2 * cell;
  const oy = bounds.height / 2 - data.config.height / 2 * cell;
  await page.locator('#world').click({ position: { x: ox + (body.x + .5) * cell, y: oy + (body.y + .5) * cell } });
  await expect(page.locator('#organism')).toContainText(`Organism ${body.id}`);
  return body;
}

test('live controls, keyboard stepping, inspector, graph and screenshots', async ({ page, request }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.locator('#step').click();
  await expect(page.locator('#tick')).toContainText('tick 1 /');
  await page.locator('#play').click();
  await expect(page.locator('#play')).toHaveText('Ⅱ Pause');
  await page.locator('#play').click();
  const first = await (await request.get('/api/state')).json();
  await page.waitForTimeout(150);
  const second = await (await request.get('/api/state')).json();
  expect(second.frame.tick).toBe(first.frame.tick);
  await page.locator('#fit').click();
  await pickOrganism(page, request);
  await page.locator('#follow').click();
  await expect(page.locator('#follow')).toHaveText('Following ✓');
  await page.locator('#grid').click();
  await expect(page.locator('#grid')).toHaveAttribute('aria-pressed', 'true');
  await page.locator('#layer').selectOption('energy');
  await fs.mkdir('docs/studio', { recursive: true });
  await page.screenshot({ path: 'docs/studio/ecosystem.png', fullPage: true });
  await page.locator('#open-neural').click();
  await expect(page.locator('#network .node')).toHaveCount(13);
  await page.locator('#network .node').first().click();
  await expect(page.locator('#network-detail')).toContainText('activation');
  await page.locator('#genome-compare').selectOption('0:1');
  await expect(page.locator('#network-compare-card')).toBeVisible();
  await page.screenshot({ path: 'docs/studio/neural.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('configuration validation, frozen definitions and browser presets', async ({ page, request }) => {
  await page.locator('#configure').click();
  await page.locator('#config-population').fill('200');
  await page.getByRole('button', { name: 'Start configured experiment →' }).click();
  await expect(page.locator('#config-error')).toContainText('fit in the world');
  await page.locator('#preset').selectOption('scarce');
  await page.locator('#config-ticks').fill('15');
  await page.locator('#config-generations').fill('1');
  await page.locator('#preset-save').click();
  await page.locator('#config-food').fill('1');
  await page.locator('#preset-load').click();
  await expect(page.locator('#config-food')).toHaveValue('30');
  await page.screenshot({ path: 'docs/studio/laboratory.png', fullPage: true });
  await page.getByRole('button', { name: 'Start configured experiment →' }).click();
  await expect(page.locator('#config-dialog')).not.toBeVisible();
  const state = await (await request.get('/api/state')).json();
  expect(state.config.food).toBe(30);
  expect(state.paused).toBe(true);
});

test('complete evolution, replay scrub, generation navigation and exports', async ({ page, request }) => {
  for (let tick = 0; tick < 25; tick++) await request.post('/api/control', { data: { action: 'step' } });
  await expect(page.locator('#mode')).toContainText('COMPLETE');
  await page.locator('[data-view=evolution]').click();
  await expect(page.locator('#history tbody tr')).toHaveCount(2);
  await page.screenshot({ path: 'docs/studio/evolution.png', fullPage: true });
  await page.locator('[data-view=ecosystem]').click();
  await page.locator('#review').click();
  await expect(page.locator('#timeline')).toBeVisible();
  await page.locator('#scrub').fill('4');
  await expect(page.locator('#replay-position')).toContainText('G0:T4');
  await page.locator('#generation-jump').selectOption({ label: 'Generation 1' });
  await expect(page.locator('#replay-position')).toContainText('G1:T0');
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloadPromise;
  const replayPath = 'test-results/replay.json';
  await download.saveAs(replayPath);
  const recorded = JSON.parse(await fs.readFile(replayPath, 'utf8'));
  expect(recorded.frames).toHaveLength(26);
  await page.locator('#live').click();
  await page.locator('#import').setInputFiles(replayPath);
  await expect(page.locator('#replay-position')).toContainText('Frame 1/26');
  await page.locator('#compare-import').setInputFiles(replayPath);
  await expect(page.locator('#comparison-panel')).toBeVisible();
  const csvPromise = page.waitForEvent('download');
  await page.locator('#csv').click();
  expect((await csvPromise).suggestedFilename()).toBe('clage-metrics.csv');
  const pngPromise = page.waitForEvent('download');
  await page.locator('#screenshot').click();
  expect((await pngPromise).suggestedFilename()).toBe('clage-world.png');
  await page.locator('#clear-compare').click();
  await expect(page.locator('#comparison-panel')).not.toBeVisible();
});

test('research baselines, local save and responsive view', async ({ page, request }) => {
  for (let tick = 0; tick < 12; tick++) await request.post('/api/control', { data: { action: 'step' } });
  await page.locator('[data-view=research]').click();
  await page.locator('#evaluate').click();
  await expect(page.locator('#evaluation tbody tr')).toHaveCount(20);
  await expect(page.locator('#evaluation')).toContainText('sample SD');
  await page.screenshot({ path: 'docs/studio/research.png', fullPage: true });
  await page.locator('#save').click();
  await expect(page.locator('#toast')).toContainText('Saved');
  await expect(page.locator('#artifacts a').first()).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-view=ecosystem]').click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: 'docs/studio/mobile.png', fullPage: true });
});

test('large population, empty food and error handling', async ({ page, request }) => {
  await request.post('/api/runs', { data: { ...config, population: 1000, width: 64, height: 64, food: 0, ticks: 3, generations: 1 } });
  await page.reload();
  await expect(page.locator('#render-rate')).not.toContainText('—');
  await expect(page.locator('#population')).toHaveText('1,000');
  await expect(page.locator('#food')).toHaveText('0');
  await page.locator('#step').click();
  await expect(page.locator('#tick')).toContainText('tick 1 /');
  await page.locator('#layer').selectOption('density');
  await page.screenshot({ path: 'docs/studio/large-population.png', fullPage: true });
  await page.locator('#import').setInputFiles({ name: 'invalid.json', mimeType: 'application/json', buffer: Buffer.from('{"schema":"no"}') });
  await expect(page.locator('#toast')).toContainText('Unsupported');
});
