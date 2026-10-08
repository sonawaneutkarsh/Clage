import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

const config = { population: 12, generations: 2, ticks: 500, width: 16, height: 12, food: 30, seed: 42 };
const evidence = 'docs/final-review/2026-10-08';

test.beforeEach(async ({ page, request }) => {
  page.on('pageerror', error => { throw error; });
  await request.post('/api/runs', { data: config });
  await page.goto('/');
  await expect(page.locator('#population')).toHaveText('12');
});

test('nested inspectors return in order with selection, follow, camera and settings intact', async ({ page }) => {
  await expect(page.locator('#view-back')).toBeHidden();
  await page.locator('#body-choice').selectOption('0');
  await expect(page.locator('#organism')).toContainText('Organism 0');
  await page.locator('#follow').click();
  await page.locator('#layer').selectOption('energy');
  await page.locator('#grid').click();
  await page.locator('#speed').fill('10');
  await page.locator('#step').click();
  await expect(page.locator('#tick')).toContainText('tick 1 /');
  const tick = await page.locator('#tick').textContent();
  await page.locator('#open-neural').click();
  await expect(page.locator('#view-back')).toHaveText('← Back to organism inspector');
  const node = page.locator('#network .node').first();
  await node.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#network-detail')).toContainText('"kind": "Node"');
  await expect(page.locator('#view-back')).toHaveText('← Back to network');
  await page.keyboard.press('Escape');
  await expect(page.locator('#view-back')).toHaveText('← Back to organism inspector');
  await page.locator('#view-back').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#ecosystem-view')).toBeVisible();
  await expect(page.locator('#organism')).toContainText('Organism 0');
  await expect(page.locator('#follow')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#layer')).toHaveValue('energy');
  await expect(page.locator('#grid')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#tick')).toHaveText(tick);
  await expect(page.locator('#open-neural')).toBeFocused();
  await page.goForward();
  await expect(page.locator('#neural-view')).toBeVisible();
  await page.goBack();
  await page.locator('#view-back').click();
  await expect(page.locator('#view-back')).toBeHidden();
  await expect(page.locator('#body-choice')).toHaveValue('');
  await page.goForward();
  await expect(page.locator('#body-choice')).toHaveValue('0');
  await expect(page.locator('#view-back')).toHaveText('← Back to ecosystem');
});

test('top-level navigation, brand and browser history never reload the simulation', async ({ page, request }) => {
  await page.locator('#step').click();
  const initial = await (await request.get('/api/state')).json();
  await page.locator('[data-view=research]').click();
  await expect(page.locator('#view-back')).toBeHidden();
  await expect(page.locator('[data-view=research]')).toHaveAttribute('aria-current', 'page');
  const length = await page.evaluate(() => history.length);
  await page.locator('[data-view=research]').click();
  expect(await page.evaluate(() => history.length)).toBe(length);
  await page.locator('[data-view=evolution]').click();
  await page.goBack();
  await expect(page.locator('#research-view')).toBeVisible();
  await page.goForward();
  await expect(page.locator('#evolution-view')).toBeVisible();
  await page.locator('.brand').click();
  await expect(page.locator('#ecosystem-view')).toBeVisible();
  const final = await (await request.get('/api/state')).json();
  expect(final.run_id).toBe(initial.run_id);
  expect(final.frame).toEqual(initial.frame);
});

test('dialogs have consistent Back and Escape, including browser forward and mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-view=evolution]').click();
  await page.locator('#heading-configure').click();
  await expect(page.locator('#close-config')).toHaveText('← Back');
  await page.locator('#config-seed').fill('123');
  await page.goBack();
  await expect(page.locator('#config-dialog')).not.toBeVisible();
  await expect(page.locator('#evolution-view')).toBeVisible();
  await page.goForward();
  await expect(page.locator('#config-dialog')).toBeVisible();
  await expect(page.locator('#config-seed')).toHaveValue('123');
  await page.keyboard.press('Escape');
  await expect(page.locator('#config-dialog')).not.toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.locator('#shortcuts').click();
  await page.locator('#close-help').click();
  await expect(page.locator('#help-dialog')).not.toBeVisible();
  await expect(page.locator('#evolution-view')).toBeVisible();
});

test('champion and ancestry paths return to the exact evolutionary focus', async ({ page, request }) => {
  await request.post('/api/runs', { data: { ...config, ticks: 3, generations: 3 } });
  for (let tick = 0; tick < 11; tick++) await request.post('/api/control', { data: { action: 'step' } });
  await page.reload();
  await page.locator('[data-view=evolution]').click();
  await expect(page.locator('#history tbody tr')).toHaveCount(3);
  await page.locator('#history button').filter({ hasText: 'Ancestry' }).last().click();
  const focus = await page.locator('#evolution-genome').inputValue();
  const parents = page.locator('#evolution-tree .genome-ancestor');
  expect(await parents.count()).toBeGreaterThan(1);
  await parents.nth(1).focus();
  await page.keyboard.press('Space');
  await expect(page.locator('#view-back')).toHaveText('← Back to ancestry');
  await page.locator('#view-back').click();
  await expect(page.locator('#evolution-genome')).toHaveValue(focus);
  await page.locator('#history button').filter({ hasText: 'Inspect' }).first().click();
  await expect(page.locator('#neural-view')).toBeVisible();
  await expect(page.locator('#view-back')).toHaveText('← Back to evolution');
  await page.locator('#view-back').click();
  await expect(page.locator('#evolution-view')).toBeVisible();
  await expect(page.locator('#evolution-genome')).toHaveValue(focus);
});

test('replay return restores live inspection and clears comparison legend', async ({ page, request }) => {
  await page.locator('#body-choice').selectOption('0');
  await page.locator('#follow').click();
  await page.locator('#step').click();
  await page.locator('#review').click();
  await expect(page.locator('#mode')).toContainText('REPLAY');
  await page.locator('#speed').fill('120');
  const replay = await (await request.get('/api/replay')).body();
  await page.locator('#compare-import').setInputFiles({ name: 'compare.json', mimeType: 'application/json', buffer: replay });
  await expect(page.locator('.compare-key')).toBeVisible();
  await page.locator('#live').click();
  await expect(page.locator('#organism')).toContainText('Organism 0');
  await expect(page.locator('#follow')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.compare-key')).toBeHidden();
  await expect(page.locator('#comparison-panel')).toBeHidden();
  const live = await (await request.get('/api/state')).json();
  await expect(page.locator('#speed')).toHaveValue(String(live.speed));
});

test('history cannot resurrect a previous run body or stale ancestry metadata', async ({ page, request }) => {
  await page.locator('#body-choice').selectOption('0');
  await page.locator('#open-neural').click();
  await request.post('/api/runs', { data: { ...config, seed: 99 } });
  await expect(page.locator('#world-seed')).toContainText('99');
  await page.goBack();
  await expect(page.locator('#body-choice')).toHaveValue('');
  await page.locator('[data-view=evolution]').click();
  await expect(page.locator('#evolution-detail')).not.toContainText('"generation": 1');
});

test('navigation preserves a running worker and final-review screens at multiple sizes', async ({ page }) => {
  await fs.mkdir(evidence, { recursive: true });
  await page.locator('#body-choice').selectOption('0');
  await page.locator('#speed').fill('10');
  await page.locator('#speed').dispatchEvent('change');
  await page.locator('#play').click();
  await page.locator('#open-neural').click();
  await expect(page.locator('#mode')).toContainText('RUNNING');
  await page.locator('#view-back').click();
  await expect(page.locator('#mode')).toContainText('RUNNING');
  await page.locator('#play').click();
  await expect(page.locator('#render-rate')).not.toContainText('—');
  await page.screenshot({ path: `${evidence}/ecosystem.png`, fullPage: true });
  await page.locator('#open-neural').click();
  await page.screenshot({ path: `${evidence}/neural-back.png`, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: `${evidence}/mobile-neural-back.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.locator('#view-back').click();
  await page.screenshot({ path: `${evidence}/mobile-ecosystem.png`, fullPage: true });
});

test('keyboard graph inspection does not trigger playback and hash reload opens a top-level view', async ({ page, request }) => {
  await page.locator('[data-view=neural]').click();
  await page.locator('#network .node').first().focus();
  await page.keyboard.press('Space');
  await expect(page.locator('#network-detail')).toContainText('"kind": "Node"');
  expect((await (await request.get('/api/state')).json()).paused).toBe(true);
  await page.locator('#genome-choice').selectOption({ index: 1 });
  await expect(page.locator('#network-detail')).not.toContainText('"kind": "Node"');
  await page.reload();
  await expect(page.locator('#neural-view')).toBeVisible();
  await expect(page.locator('#network .node')).toHaveCount(13);
  await expect(page.locator('#view-back')).toBeHidden();
});

test('small-screen evolution, research, validation and comparison remain usable', async ({ page, request }) => {
  await fs.mkdir(evidence, { recursive: true });
  await request.post('/api/runs', { data: { ...config, ticks: 3 } });
  for (let tick = 0; tick < 7; tick++) await request.post('/api/control', { data: { action: 'step' } });
  await page.reload();
  await page.setViewportSize({ width: 390, height: 844 });
  for (const view of ['evolution', 'research']) {
    await page.locator(`[data-view=${view}]`).click();
    await expect(page.locator(`#${view}-view`)).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    await page.screenshot({ path: `${evidence}/mobile-${view}.png`, fullPage: true });
  }
  await page.locator('#heading-configure').click();
  await page.locator('#config-food').fill('999');
  await page.locator('#config-form [type=submit]').click();
  await expect(page.locator('#config-error')).not.toBeEmpty();
  await page.screenshot({ path: `${evidence}/configuration-error.png`, fullPage: true });
  await page.locator('#close-config').click();
  await expect(page.locator('#research-view')).toBeVisible();
  await page.locator('[data-view=ecosystem]').click();
  await page.locator('#review').click();
  const replay = await (await request.get('/api/replay')).body();
  await page.locator('#compare-import').setInputFiles({ name: 'comparison.json', mimeType: 'application/json', buffer: replay });
  await page.locator('#scrub').fill('4');
  await page.locator('#scrub').dispatchEvent('input');
  await expect(page.locator('#replay-position')).toContainText('Frame 5/8');
  await expect(page.locator('#comparison-panel')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${evidence}/mobile-replay-comparison.png`, fullPage: true });
});

test('starting a configured run clears stale replay comparison controls and old selections', async ({ page, request }) => {
  await page.locator('#body-choice').selectOption('0');
  await page.locator('#step').click();
  await expect(page.locator('#tick')).toContainText('tick 1 /');
  await page.locator('#review').click();
  const replay = await (await request.get('/api/replay')).body();
  await page.locator('#compare-import').setInputFiles({ name: 'comparison.json', mimeType: 'application/json', buffer: replay });
  await expect(page.locator('.compare-key')).toBeVisible();
  await page.locator('#heading-configure').click();
  await page.locator('#config-seed').fill('100');
  await page.locator('#config-form [type=submit]').click();
  await expect(page.locator('#config-dialog')).not.toBeVisible();
  await expect(page.locator('#world-seed')).toContainText('100');
  await expect(page.locator('#mode')).toContainText('LIVE · PAUSED');
  for (const selector of ['.compare-key', '#clear-compare', '#comparison-panel', '#timeline', '#view-back']) {
    await expect(page.locator(selector)).toBeHidden();
  }
  await page.goBack();
  await expect(page.locator('#body-choice')).toHaveValue('');
});

test('world canvas fills its viewport on desktop and small screens', async ({ page }) => {
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    const dimensions = await page.evaluate(() => {
      const canvas = document.getElementById('world'), wrap = document.getElementById('canvas-wrap');
      return { canvas: [canvas.clientWidth, canvas.clientHeight], wrap: [wrap.clientWidth, wrap.clientHeight] };
    });
    expect(dimensions.canvas).toEqual(dimensions.wrap);
    expect(dimensions.canvas[1]).toBeGreaterThanOrEqual(360);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  }
});
