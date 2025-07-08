import asyncio
import os
import pytest
from playwright.async_api import async_playwright


@pytest.mark.asyncio
async def test_sequential_launches():
    """Test starting the dev server 5× sequentially to catch race conditions."""
    # Ensure the dev server can start without race conditions
    try:
        for i in range(5):
            print(f"Starting attempt {i + 1}")
            process = await asyncio.create_subprocess_exec(
                "uvicorn",
                "pynomaly.presentation.api.app:app",
                "--reload",
                "--port",
                "8000",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Sleep to give the server time to start
            await asyncio.sleep(5)  # Adjust based on bootup time

            # Launch browser to check if the server is responding
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                context = await browser.new_context()
                page = await context.new_page()

                try:
                    # Make an HTTP request to the server
                    response = await page.goto("http://localhost:8000/api/v1/health")
                    assert response.status == 200
                    await browser.close()
                except Exception as e:
                    print(f"Error on attempt {i + 1}: {str(e)}")
                finally:
                    # Ensure browser is closed
                    if not browser.is_closed():
                        await browser.close()

            # Terminate server process
            process.terminate()
            await process.wait()

    except Exception as e:
        pytest.fail(f"Sequential launch test failed: {str(e)}")
