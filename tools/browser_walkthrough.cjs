/* Real browser checks and optional screen recording of the running local app. */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const playwright = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const out = {started_at: new Date().toISOString(), checks: [], events: []};
  const browser = await playwright.chromium.launch({
    executablePath: process.env.CHROME_BIN || undefined,
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']
  });
  const context = await browser.newContext({viewport: {width: 1440, height: 1100},
    ...(spec.record ? {recordVideo: {dir: spec.video_directory,
      size: {width: 1440, height: 1100}}} : {})});
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(String(error)));
  const video = spec.record ? page.video() : null;
  const started = Date.now();
  async function caption(title, body, delay = 6500) {
    if (!spec.record) return;
    await page.evaluate(({title, body}) => {
      let element = document.getElementById('recording-caption');
      if (!element) {
        element = document.createElement('aside'); element.id = 'recording-caption';
        element.style.cssText = 'position:fixed;bottom:16px;left:22px;right:22px;z-index:1000;background:#123c32f5;color:white;border-radius:12px;padding:15px 22px;font:17px/1.5 Arial;box-shadow:0 5px 25px #0003;pointer-events:none';
        document.body.appendChild(element);
      }
      element.replaceChildren();
      const heading = document.createElement('strong'); heading.textContent = title;
      const description = document.createElement('div'); description.textContent = body;
      element.append(heading, description);
    }, {title, body});
    out.events.push({seconds: (Date.now() - started)/1000, title, body});
    await page.waitForTimeout(delay);
  }
  try {
    await page.goto(spec.url, {waitUntil: 'networkidle'});
    await page.locator('#status').filter({hasText: 'Trained model ready'}).waitFor();
    assert(await page.locator('#development').isHidden());
    assert(await page.locator('#inspect').isDisabled());
    out.checks.push('Real model ready; development banner hidden; empty input disabled');
    await caption('Risk-aware, explainable fabric inspection',
      'Actual screen recording of the local application. A trained RAW-FABRID model performs a fresh prediction for each upload. This walkthrough is scripted with browser automation.');
    for (let index = 0; index < spec.cases.length; index++) {
      const item = spec.cases[index];
      await page.locator('#file').setInputFiles(item.path);
      await caption(`Case ${index + 1}: ${item.label === 1 ? 'annotated defective' : 'annotated defect-free'} patch`,
        `The publisher annotation is ${item.label === 1 ? 'defective' : 'defect-free'}. Ground truth is supplied only for this demonstration; the application receives the image pixels.`);
      const promise = page.waitForResponse(response => response.url().endsWith('/explain') && response.request().method() === 'POST');
      await page.locator('#inspect').click();
      const response = await promise;
      assert(response.ok()); const result = await response.json();
      await page.locator('#result-title').filter({hasText: 'Patch inspected'}).waitFor();
      assert.equal(result.provenance, 'RAW-FABRID_V1');
      assert.equal(result.model_version, spec.model_version);
      assert(Math.abs(result.defect_score-item.expected_score) < 1e-5);
      assert.equal(await page.locator('#prediction').textContent(), result.class === 'defective' ? 'Defective' : 'Defect-free');
      assert((await page.locator('#score-kind').textContent()).includes('Uncalibrated'));
      await page.waitForFunction(() => { const image = document.getElementById('overlay'); return image.complete && image.naturalWidth === 256; });
      const wrong = Number(result.class === 'defective') !== item.label;
      out.checks.push(`Fresh UI inference and Grad-CAM parity: case ${index+1}`);
      out.events.push({seconds: (Date.now()-started)/1000, case_index:index,
        patch_id:item.patch_id, label:item.label, prediction:result.class,
        score:result.defect_score, incorrect:wrong, model_version:result.model_version});
      await page.locator('#results').scrollIntoViewIfNeeded();
      await caption(wrong ? 'A genuine incorrect prediction' : 'Prediction and score',
        `The model predicts ${result.class === 'defective' ? 'defective' : 'defect-free'} with score ${(result.defect_score*100).toFixed(1)}%. ${wrong ? 'This disagrees with the publisher annotation. The incorrect prediction is retained and analysed in the report.' : 'The score is not guaranteed confidence. The classification cutoff was selected on validation data.'}`, 8500);
      await page.locator('#opacity').evaluate(element => {element.value='0';element.dispatchEvent(new Event('input', {bubbles:true}));});
      assert.equal(await page.locator('#opacity-value').textContent(), '0%');
      await caption('Original pixels and the explanation',
        'Opacity is zero: inspect the original patch. The original image remains available alongside the explanation.');
      await page.locator('#opacity').evaluate(element => {element.value='75';element.dispatchEvent(new Event('input', {bubbles:true}));});
      assert.equal(await page.locator('#opacity-value').textContent(), '75%');
      await caption('Grad-CAM of the defect logit',
        result.explanation.zero_attribution ? 'This case has no positive attribution. The application reports the zero map without inventing a hotspot.' : 'Bright regions show positive Grad-CAM attribution for the defect logit. Occlusion checks in the report test sensitivity; this is not a certified defect boundary or a causal explanation.', 8000);
      if (index === 0 && spec.screenshot) await page.screenshot({path:spec.screenshot, fullPage:true});
      await page.locator('#opacity').evaluate(element => {element.value='55';element.dispatchEvent(new Event('input', {bubbles:true}));});
    }
    await page.locator('details').locator('summary').click();
    await page.locator('details').scrollIntoViewIfNeeded();
    await caption('Human review and limitations',
      'Inspection bands are validation-selected decision rules with assumed costs. A patch cannot certify a roll. Unseen fabrics, changed lighting, blur and tiny defects can cause errors.');
    await page.locator('#file').setInputFiles({name:'invalid.png', mimeType:'image/png', buffer:Buffer.from('not an image')});
    await page.locator('#inspect').click();
    await page.locator('#status.error').filter({hasText:'Invalid or unsafe image content'}).waitFor();
    out.checks.push('Invalid input rejected with visible error; UI remains usable');
    await caption('Invalid inputs are rejected',
      'This upload contains invalid image bytes. The server rejects it and the interface displays the error. Valid input must be a 256 x 256 PNG or JPEG of at most 5 MiB.');
    assert.deepEqual(pageErrors, []);
    out.checks.push('No browser JavaScript errors');
    if (spec.record) assert(out.events.some(event => event.incorrect === true));
    await caption('Evidence and reproducibility',
      'The report contains the frozen split, actual experiment logs, a shared held-out comparison, and three incorrect predictions analysed with XAI. Model limitations and AI assistance are declared.', 7500);
    await caption('Dataset attribution',
      'RAW-FABRID V1: R. Perez-Llorens, T. Albero-Albero and J. Silvestre-Blanes (2026), Mendeley Data, DOI 10.17632/db6g85xsyg.1, CC BY 4.0. This recorded walkthrough uses original publisher patches.');
    out.status='PASS';
  } finally {
    await context.close();
    if (video) out.webm_path = await video.path();
    await browser.close();
    out.finished_at = new Date().toISOString();
    out.wall_seconds = (Date.now()-started)/1000;
    fs.writeFileSync(spec.result_path, JSON.stringify(out, null, 2)+'\n');
  }
}
main().catch(error => {console.error(error);process.exitCode=1;});
