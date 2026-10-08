import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

const config = { population: 12, generations: 2, ticks: 12, width: 16, height: 12, food: 30, regrowth: 1, seed: 42 };
let pageErrors = [];

test.beforeEach(async ({ request, page }) => {
  pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await request.post('/api/runs', { data: config });
  await page.goto('/');
  await expect(page.locator('#population')).not.toHaveText('—');
  await expect(page.locator('#render-rate')).not.toContainText('—');
});

test.afterEach(async () => expect(pageErrors).toEqual([]));

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
  await page.keyboard.press('.');
  await expect(page.locator('#tick')).toContainText('tick 2 /');
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
  await page.locator('#fit').click();
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
  await page.locator('#evolution-genome').selectOption('1:0');
  await expect(page.locator('#evolution-tree .genome-ancestor')).not.toHaveCount(0);
  await expect(page.locator('#evolution-detail')).toContainText('parents');
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
  await expect(page.locator('#import')).toHaveValue('');
  await expect(page.locator('#replay-position')).toContainText('Frame 1/26');
  await page.locator('#compare-import').setInputFiles(replayPath);
  await expect(page.locator('#comparison-panel')).toBeVisible();
  const chartPromise = page.waitForEvent('download');
  await page.locator('#chart-export').click();
  expect((await chartPromise).suggestedFilename()).toBe('clage-chart.png');
  const csvPromise = page.waitForEvent('download');
  await page.locator('#csv').click();
  expect((await csvPromise).suggestedFilename()).toBe('clage-metrics.csv');
  const pngPromise = page.waitForEvent('download');
  await page.locator('#screenshot').click();
  expect((await pngPromise).suggestedFilename()).toBe('clage-world.png');
  await page.locator('#clear-compare').click();
  await expect(page.locator('#comparison-panel')).not.toBeVisible();
  const compressed = await request.get('/api/export');
  expect(compressed.ok()).toBe(true);
  await page.locator('#import').setInputFiles({ name: 'replay.json.gz', mimeType: 'application/gzip', buffer: await compressed.body() });
  await expect(page.locator('#import')).toHaveValue('');
  await expect(page.locator('#replay-position')).toContainText('Frame 1/26');
  await page.locator('#speed').fill('10');
  await page.locator('#play').click();
  await expect(page.locator('#mode')).toContainText('REPLAY · PLAYING');
  await expect(page.locator('#replay-position')).not.toContainText('Frame 1/26');
  await page.locator('#play').click();
  await expect(page.locator('#mode')).toContainText('REPLAY · PAUSED');
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
  await page.setViewportSize({ width: 1024, height: 768 });
  await page.locator('[data-view=ecosystem]').click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1024);
  await page.screenshot({ path: 'docs/studio/tablet.png', fullPage: true });
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
  await request.post('/api/runs', { data: { ...config, food: 0, metabolism: 1.0, repro_threshold: 2.0 } });
  await expect(page.locator('#population')).toHaveText('12');
  await page.locator('#step').click();
  await expect(page.locator('#population')).toHaveText('0');
  await expect(page.locator('#energy')).toHaveText('0.000');
  await page.screenshot({ path: 'docs/studio/empty-world.png', fullPage: true });
});

test('background multi-generation run stays interactive and finishes with evaluated history', async ({ page, request }) => {
  await request.post('/api/runs', { data: { population: 32, generations: 3, ticks: 150,
    width: 24, height: 24, food: 80, seed: 55 } });
  await expect(page.locator('#population')).toHaveText('32');
  await page.locator('#speed').fill('120');
  await page.locator('#speed').dispatchEvent('change');
  await page.locator('#play').click();
  await page.locator('#layer').selectOption('food');
  await expect(page.locator('#mode')).toContainText('RUNNING');
  await page.locator('[data-view=neural]').click();
  await expect(page.locator('#network .node')).toHaveCount(13);
  await page.locator('[data-view=ecosystem]').click();
  await expect(page.locator('#mode')).toContainText('COMPLETE', { timeout: 20000 });
  await expect(page.locator('#generation')).toHaveText('02');
  await page.locator('[data-view=evolution]').click();
  await expect(page.locator('#history tbody tr')).toHaveCount(3);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#champion-export').click();
  const downloaded = await downloadPromise;
  const filename = 'test-results/champion.json';
  await downloaded.saveAs(filename);
  const champion = JSON.parse(await fs.readFile(filename, 'utf8'));
  expect(champion.genome.fitness).not.toBeNull();
  expect(champion.metadata.checkpoint).toBe(false);
});
