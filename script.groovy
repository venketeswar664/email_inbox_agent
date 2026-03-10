/* ============================================================
   script.groovy — Reusable Pipeline Helper Functions
   ============================================================
   Loaded by Jenkinsfile with:
       gv = load "script.groovy"

   Then called as:
       gv.buildApp()
       gv.testApp()
       gv.lintApp()
       gv.deployApp()

   Why a separate .groovy file?
   → Keeps Jenkinsfile clean (just orchestration/stages)
   → Functions here can be unit-tested independently
   → Easy to swap deploy strategy without touching Jenkinsfile
   ============================================================ */


// ── 1. Build ──────────────────────────────────────────────────────────────────
// Creates a Python virtual environment and installs all dependencies.
// Both runtime (requirements.txt) and dev/test (requirements-dev.txt) are installed.
def buildApp() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " BUILD — Setting up Python environment"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sh """
        echo "[1/3] Creating virtual environment..."
        ${env.PYTHON} -m venv ${env.VENV_DIR}

        echo "[2/3] Installing runtime dependencies..."
        ${env.VENV_DIR}/bin/pip install --upgrade pip --quiet
        if [ -f requirements.txt ]; then
            ${env.VENV_DIR}/bin/pip install -r requirements.txt --quiet
        fi

        echo "[3/3] Installing dev/test dependencies..."
        if [ -f requirements-dev.txt ]; then
            ${env.VENV_DIR}/bin/pip install -r requirements-dev.txt --quiet
        fi

        echo "BUILD COMPLETE ✅"
    """
}


// ── 2. Test ───────────────────────────────────────────────────────────────────
// Runs the full pytest test suite with:
//   - JUnit XML output  → Jenkins shows per-test results in the UI
//   - HTML coverage      → published as 'Coverage Report' in Jenkins
//   - Minimum coverage   → pipeline fails if coverage < 60%
def testApp() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " TEST — Running Unit Tests (Version: ${params.VERSION})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sh """
        echo "Running pytest on tests/ ..."

        ${env.VENV_DIR}/bin/pytest tests/ \\
            --tb=short \\
            -v \\
            --junitxml=test-results.xml \\
            --cov=. \\
            --cov-report=html:htmlcov \\
            --cov-report=term-missing \\
            --cov-fail-under=60

        echo "TESTS PASSED ✅"
    """
}


// ── 3. Lint ───────────────────────────────────────────────────────────────────
// Runs flake8 on all main Python source files.
// Max line length set to 120 (matches the docstrings in model_server.py).
// Files excluded: generated files, venv, tests, __pycache__.
def lintApp() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " LINT — Checking code style (flake8)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sh """
        ${env.VENV_DIR}/bin/flake8 \\
            main.py orchestrator.py model_server.py generate_token.py \\
            --max-line-length=120 \\
            --exclude=${env.VENV_DIR},__pycache__,*.pyc \\
            --statistics

        echo "LINT PASSED ✅"
    """
}


// ── 4. Deploy ─────────────────────────────────────────────────────────────────
// Restarts both the Gmail agent (port 8000) and the MCP model server (port 8011).
// METHOD: Uses pkill to stop existing uvicorn processes, then relaunches them
// in the background using nohup so they survive the Jenkins agent session.
//
// ⚙️  To adapt this for your setup:
//   - systemd:  replace sh block with "sudo systemctl restart email-agent"
//   - Docker:   replace with "docker-compose up -d"
//   - Remote:   wrap in sshagent { sh "ssh user@host 'systemctl restart ...'" }
def deployApp() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " DEPLOY — Deploying Version: ${params.VERSION}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sh """
        APP_DIR=\$(pwd)

        echo "Stopping existing Gmail agent (port ${env.APP_PORT})..."
        pkill -f "uvicorn main:app" || true     # 'true' so we don't fail if nothing was running

        echo "Stopping existing MCP server (port ${env.MCP_PORT})..."
        pkill -f "uvicorn model_server:app" || true

        sleep 2   # Give processes time to fully terminate

        echo "Starting MCP Model Server on port ${env.MCP_PORT} (version ${params.VERSION})..."
        cd \$APP_DIR
        nohup ${env.VENV_DIR}/bin/uvicorn model_server:app \\
            --host 0.0.0.0 \\
            --port ${env.MCP_PORT} \\
            > logs/mcp_server.log 2>&1 &

        # Wait for MCP to be ready before starting the agent
        echo "Waiting for MCP server to become healthy..."
        for i in \$(seq 1 15); do
            STATUS=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${env.MCP_PORT}/health || echo "000")
            if [ "\$STATUS" = "200" ]; then
                echo "MCP server is healthy ✅"
                break
            fi
            echo "  Attempt \$i/15... (MCP returned \$STATUS)"
            sleep 3
        done

        echo "Starting Gmail Agent on port ${env.APP_PORT} (version ${params.VERSION})..."
        nohup ${env.VENV_DIR}/bin/uvicorn main:app \\
            --host 0.0.0.0 \\
            --port ${env.APP_PORT} \\
            > logs/gmail_agent.log 2>&1 &

        echo "Both services launched!"
        echo ""
        echo "  Gmail Agent  → http://localhost:${env.APP_PORT}/health"
        echo "  MCP Server   → http://localhost:${env.MCP_PORT}/health"
        echo "  Logs dir     → \$APP_DIR/logs/"
        echo ""
        echo "DEPLOY COMPLETE ✅ (version ${params.VERSION})"
    """
}

// Required by Jenkins when loading a Groovy script
return this
