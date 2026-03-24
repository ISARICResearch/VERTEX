import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import pytest
import requests

pytestmark = pytest.mark.e2e

playwright_sync_api = pytest.importorskip("playwright.sync_api")
Error = playwright_sync_api.Error
sync_playwright = playwright_sync_api.sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
ANALYSIS_PROJECT_ID = "archetypecrf-mpox-synthetic"
PREBUILT_PROJECT_ID = "prebuilt-public-fixture"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def prebuilt_projects_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("playwright-projects")
    shutil.copytree(FIXTURES_ROOT / "prebuilt_public_project", root / "prebuilt_public_project")
    return root


@pytest.fixture(scope="module")
def vertex_server(prebuilt_projects_root):
    port = _free_port()
    env = os.environ.copy()
    env.pop("APP_ENV", None)
    env["VERTEX_PROJECTS_DIR"] = str(prebuilt_projects_root)
    env["PYTHONPATH"] = str(REPO_ROOT)

    cmd = [
        sys.executable,
        "-c",
        (
            "from vertex.descriptive_dashboard import app; "
            f"app.run(host='127.0.0.1', port={port}, debug=False, use_reloader=False)"
        ),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"
    start = time.time()
    response_error = None
    while time.time() - start < 30:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"VERTEX server exited during startup.\n{output}")
        try:
            response = requests.get(base_url, timeout=1)
            if response.ok:
                break
        except requests.RequestException as exc:
            response_error = exc
            time.sleep(0.5)
    else:
        output = process.stdout.read() if process.stdout else ""
        raise RuntimeError(f"Timed out waiting for VERTEX server: {response_error}\n{output}")

    yield base_url

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.fixture()
def page():
    with sync_playwright() as playwright_context:
        try:
            browser = playwright_context.chromium.launch(headless=True)
        except Error as exc:
            pytest.skip(f"Playwright Chromium browser unavailable: {exc}")
        page = browser.new_page()
        try:
            yield page
        finally:
            browser.close()


def test_analysis_project_loads(vertex_server, page):
    page.goto(f"{vertex_server}/?project={quote_plus(ANALYSIS_PROJECT_ID)}")

    page.get_by_role("heading", name="(Analysis) ARChetype CRF mpox").wait_for(timeout=30000)
    page.locator("#world-map .js-plotly-plot").wait_for(timeout=30000)


def _select_project(page, project_name):
    page.locator("#project-selector").click()
    page.locator("#project-selector input").fill(project_name)
    page.locator(".VirtualizedSelectOption").filter(has_text=project_name).first.click()


def test_prebuilt_project_loads(vertex_server, page):
    page.goto(vertex_server)
    _select_project(page, "Prebuilt Public Fixture")

    page.get_by_role("heading", name="Prebuilt Public Fixture").wait_for(timeout=30000)
    page.locator("#world-map .js-plotly-plot").wait_for(timeout=30000)


def test_prebuilt_project_url_param_persists(vertex_server, page):
    page.goto(f"{vertex_server}/?project={quote_plus(PREBUILT_PROJECT_ID)}")

    page.get_by_role("heading", name="Prebuilt Public Fixture").wait_for(timeout=30000)
    page.locator("#world-map .js-plotly-plot").wait_for(timeout=30000)
    page.wait_for_function(
        """
        () => {
            const heading = document.querySelector('h4');
            return heading && heading.textContent && heading.textContent.includes('Prebuilt Public Fixture');
        }
        """,
        timeout=30000,
    )


def test_prebuilt_modal_opens(vertex_server, page):
    page.goto(vertex_server)
    _select_project(page, "Prebuilt Public Fixture")

    page.get_by_text("Insights").first.click()
    page.get_by_role("button", name="Panel A").click()

    page.locator("#modal").wait_for(timeout=30000)
    page.get_by_text("Insights: Panel A").wait_for(timeout=30000)
    page.get_by_text("Test graph").wait_for(timeout=30000)


def test_analysis_filter_interaction_updates_country_display(vertex_server, page):
    page.goto(f"{vertex_server}/?project={quote_plus(ANALYSIS_PROJECT_ID)}")

    page.get_by_text("Filters and Controls").first.click()
    page.locator("#country-display").wait_for(timeout=30000)
    select_all = page.locator("#country-selectall label").first
    select_all.evaluate("(el) => el.click()")

    page.wait_for_function(
        """
        () => {
            const node = document.querySelector('#country-display');
            return node && node.textContent && node.textContent.includes('None selected');
        }
        """,
        timeout=30000,
    )
