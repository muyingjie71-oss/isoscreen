import asyncio, sys, time, os
from playwright.async_api import async_playwright
URL = os.environ.get('ISOSCREEN_URL', 'http://localhost:8765/index.html')   # serve webtool/site first
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={'width': 1280, 'height': 1000})
        logs = []; pg.on('console', lambda m: logs.append(f'{m.type}: {m.text}')); pg.on('pageerror', lambda e: logs.append(f'PAGEERROR: {e}'))
        await pg.goto(URL); t0 = time.time()
        await pg.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')", timeout=600000)
        print('ready after', round(time.time() - t0, 1), 's')
        for ex in sys.argv[1:]:
            t0 = time.time(); await pg.click(f'[data-ex="{ex}"]')
            await pg.wait_for_function("document.querySelector('#status').textContent.startsWith('Screened')", timeout=600000)
            print('example', ex, round(time.time() - t0, 1), 's |', await pg.inner_text('#status'))
            print((await pg.inner_text('#results'))[:900].replace('\n', ' | '))
            await pg.screenshot(path=f'/tmp/ui_ex{ex}.png', full_page=True)
            await pg.evaluate("document.querySelector('#status').textContent=''")
        print('\n'.join(l for l in logs if 'error' in l.lower())[:3000])
        await b.close()
asyncio.run(main())
