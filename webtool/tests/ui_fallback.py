import asyncio, time, os
from playwright.async_api import async_playwright
URL = os.environ.get('ISOSCREEN_URL', 'http://localhost:8765/index.html')   # serve webtool/site first
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1280, "height": 900}); await pg.add_init_script("window.Worker = function(){ throw new Error(\"blocked by test\") }")
        logs = []; pg.on('console', lambda m: logs.append(f'{m.type}: {m.text}')); pg.on('pageerror', lambda e: logs.append(f'PAGEERROR: {e}'))
        await pg.goto(URL); t0 = time.time()
        await pg.wait_for_function("window.initRDKitModule && document.querySelector('#go') && !document.querySelector('#go').disabled", timeout=120000)
        await pg.click('[data-ex="0"]'); await asyncio.sleep(0.8)
        print('pending:', await pg.inner_text('#load')); await pg.screenshot(path='/tmp/ui_load.png', clip={'x': 0, 'y': 0, 'width': 1280, 'height': 560})
        await pg.wait_for_function("document.querySelector('#status').textContent.startsWith('Screened')", timeout=300000)
        print('done after', round(time.time() - t0, 1), 's |', await pg.inner_text('#status'), '| load hidden:', await pg.evaluate("document.querySelector('#load').classList.contains('hidden')"))
        print('\n'.join(l for l in logs if 'rror' in l or 'warn' in l.lower())[:2000]); await b.close()
asyncio.run(main())
