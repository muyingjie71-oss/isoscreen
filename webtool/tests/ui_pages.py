import asyncio, os
from playwright.async_api import async_playwright
URL = os.environ.get('ISOSCREEN_URL', 'http://localhost:8765/index.html')   # serve webtool/site first
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={'width': 1280, 'height': 1000})
        logs = []; pg.on('pageerror', lambda e: logs.append(f'PAGEERROR: {e}'))
        await pg.goto(URL)
        await pg.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')", timeout=300000)
        await pg.click('[data-ex="1"]'); await pg.wait_for_function("document.querySelector('#status').textContent.startsWith('Screened')", timeout=300000)
        await pg.click('tr.r'); await asyncio.sleep(1); await pg.screenshot(path='/tmp/ui_ex1.png', full_page=False)
        await pg.click('nav a[data-view=pair]'); await asyncio.sleep(1); await pg.screenshot(path='/tmp/ui_pair.png', full_page=False)
        await pg.click('nav a[data-view=method]'); await asyncio.sleep(1); await pg.screenshot(path='/tmp/ui_method.png', full_page=True)
        print('\n'.join(logs)); await b.close()
asyncio.run(main())
