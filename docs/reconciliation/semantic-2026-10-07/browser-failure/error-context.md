# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: studio.spec.js >> configuration validation, frozen definitions and browser presets
- Location: tests/browser/studio.spec.js:64:1

# Error details

```
Error: expect(locator).not.toContainText(expected) failed

Locator: locator('#render-rate')
Expected substring: not "—"
Received string: "— render FPS"
Timeout: 5000ms

Call log:
  - Expect "not toContainText" locator('#render-rate') with timeout 5000ms
  - waiting for locator('#render-rate')
    5 × locator resolved to <span id="render-rate">— render FPS</span>
      - unexpected value "— render FPS"

```

```yaml
- text: — render FPS
```

# Test source

```ts
  1   | import { test, expect } from '@playwright/test';
  2   | import fs from 'node:fs/promises';
  3   | 
  4   | const config = { population: 12, generations: 2, ticks: 12, width: 16, height: 12, food: 30, regrowth: 1, seed: 42 };
  5   | let pageErrors = [];
  6   | 
  7   | test.beforeEach(async ({ request, page }) => {
  8   |   pageErrors = [];
  9   |   page.on('pageerror', error => pageErrors.push(error.message));
  10  |   await request.post('/api/runs', { data: config });
  11  |   await page.goto('/');
  12  |   await expect(page.locator('#population')).not.toHaveText('—');
> 13  |   await expect(page.locator('#render-rate')).not.toContainText('—');
      |                                                  ^ Error: expect(locator).not.toContainText(expected) failed
  14  | });
  15  | 
  16  | test.afterEach(async () => expect(pageErrors).toEqual([]));
  17  | 
  18  | async function pickOrganism(page, request) {
  19  |   const data = await (await request.get('/api/state')).json();
  20  |   const body = data.frame.organisms.find(organism => organism.alive);
  21  |   const bounds = await page.locator('#world').boundingBox();
  22  |   const cell = Math.min((bounds.width - 75) / data.config.width, (bounds.height - 65) / data.config.height);
  23  |   const ox = bounds.width / 2 - data.config.width / 2 * cell;
  24  |   const oy = bounds.height / 2 - data.config.height / 2 * cell;
  25  |   await page.locator('#world').click({ position: { x: ox + (body.x + .5) * cell, y: oy + (body.y + .5) * cell } });
  26  |   await expect(page.locator('#organism')).toContainText(`Organism ${body.id}`);
  27  |   return body;
  28  | }
  29  | 
  30  | test('live controls, keyboard stepping, inspector, graph and screenshots', async ({ page, request }) => {
  31  |   const errors = [];
  32  |   page.on('pageerror', error => errors.push(error.message));
  33  |   await page.locator('#step').click();
  34  |   await expect(page.locator('#tick')).toContainText('tick 1 /');
  35  |   await page.keyboard.press('.');
  36  |   await expect(page.locator('#tick')).toContainText('tick 2 /');
  37  |   await page.locator('#play').click();
  38  |   await expect(page.locator('#play')).toHaveText('Ⅱ Pause');
  39  |   await page.locator('#play').click();
  40  |   const first = await (await request.get('/api/state')).json();
  41  |   await page.waitForTimeout(150);
  42  |   const second = await (await request.get('/api/state')).json();
  43  |   expect(second.frame.tick).toBe(first.frame.tick);
  44  |   await page.locator('#fit').click();
  45  |   await pickOrganism(page, request);
  46  |   await page.locator('#follow').click();
  47  |   await expect(page.locator('#follow')).toHaveText('Following ✓');
  48  |   await page.locator('#fit').click();
  49  |   await page.locator('#grid').click();
  50  |   await expect(page.locator('#grid')).toHaveAttribute('aria-pressed', 'true');
  51  |   await page.locator('#layer').selectOption('energy');
  52  |   await fs.mkdir('docs/studio', { recursive: true });
  53  |   await page.screenshot({ path: 'docs/studio/ecosystem.png', fullPage: true });
  54  |   await page.locator('#open-neural').click();
  55  |   await expect(page.locator('#network .node')).toHaveCount(13);
  56  |   await page.locator('#network .node').first().click();
  57  |   await expect(page.locator('#network-detail')).toContainText('activation');
  58  |   await page.locator('#genome-compare').selectOption('0:1');
  59  |   await expect(page.locator('#network-compare-card')).toBeVisible();
  60  |   await page.screenshot({ path: 'docs/studio/neural.png', fullPage: true });
  61  |   expect(errors).toEqual([]);
  62  | });
  63  | 
  64  | test('configuration validation, frozen definitions and browser presets', async ({ page, request }) => {
  65  |   await page.locator('#configure').click();
  66  |   await page.locator('#config-population').fill('200');
  67  |   await page.getByRole('button', { name: 'Start configured experiment →' }).click();
  68  |   await expect(page.locator('#config-error')).toContainText('fit in the world');
  69  |   await page.locator('#preset').selectOption('scarce');
  70  |   await page.locator('#config-ticks').fill('15');
  71  |   await page.locator('#config-generations').fill('1');
  72  |   await page.locator('#preset-save').click();
  73  |   await page.locator('#config-food').fill('1');
  74  |   await page.locator('#preset-load').click();
  75  |   await expect(page.locator('#config-food')).toHaveValue('30');
  76  |   await page.screenshot({ path: 'docs/studio/laboratory.png', fullPage: true });
  77  |   await page.getByRole('button', { name: 'Start configured experiment →' }).click();
  78  |   await expect(page.locator('#config-dialog')).not.toBeVisible();
  79  |   const state = await (await request.get('/api/state')).json();
  80  |   expect(state.config.food).toBe(30);
  81  |   expect(state.paused).toBe(true);
  82  | });
  83  | 
  84  | test('complete evolution, replay scrub, generation navigation and exports', async ({ page, request }) => {
  85  |   for (let tick = 0; tick < 25; tick++) await request.post('/api/control', { data: { action: 'step' } });
  86  |   await expect(page.locator('#mode')).toContainText('COMPLETE');
  87  |   await page.locator('[data-view=evolution]').click();
  88  |   await expect(page.locator('#history tbody tr')).toHaveCount(2);
  89  |   await page.locator('#evolution-genome').selectOption('1:0');
  90  |   await expect(page.locator('#evolution-tree .genome-ancestor')).not.toHaveCount(0);
  91  |   await expect(page.locator('#evolution-detail')).toContainText('parents');
  92  |   await page.screenshot({ path: 'docs/studio/evolution.png', fullPage: true });
  93  |   await page.locator('[data-view=ecosystem]').click();
  94  |   await page.locator('#review').click();
  95  |   await expect(page.locator('#timeline')).toBeVisible();
  96  |   await page.locator('#scrub').fill('4');
  97  |   await expect(page.locator('#replay-position')).toContainText('G0:T4');
  98  |   await page.locator('#generation-jump').selectOption({ label: 'Generation 1' });
  99  |   await expect(page.locator('#replay-position')).toContainText('G1:T0');
  100 |   const downloadPromise = page.waitForEvent('download');
  101 |   await page.locator('#export').click();
  102 |   const download = await downloadPromise;
  103 |   const replayPath = 'test-results/replay.json';
  104 |   await download.saveAs(replayPath);
  105 |   const recorded = JSON.parse(await fs.readFile(replayPath, 'utf8'));
  106 |   expect(recorded.frames).toHaveLength(26);
  107 |   await page.locator('#live').click();
  108 |   await page.locator('#import').setInputFiles(replayPath);
  109 |   await expect(page.locator('#import')).toHaveValue('');
  110 |   await expect(page.locator('#replay-position')).toContainText('Frame 1/26');
  111 |   await page.locator('#compare-import').setInputFiles(replayPath);
  112 |   await expect(page.locator('#comparison-panel')).toBeVisible();
  113 |   const chartPromise = page.waitForEvent('download');
```