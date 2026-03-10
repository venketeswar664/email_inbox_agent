/* ============================================================
   Jenkinsfile — Email Inbox Agent (Gmail Threat Classifier)
   ============================================================
   Pipeline structure mirrors the example you provided:
     - Global `gv` variable loads script.groovy as a reusable module
     - Parameters: VERSION choice + executeTests boolean
     - Stages: init → build → test → lint → deploy

   Pre-requisites on Jenkins agent:
     - Python 3.11+ installed
     - pip available
     - Jenkins has write access to the workspace
   ============================================================ */

def gv   // loaded from script.groovy — holds buildApp(), testApp(), etc.

pipeline {
    agent any

    // ── Build Parameters ──────────────────────────────────────────────────────
    parameters {
        choice(
            name: 'VERSION',
            choices: ['1.1.0', '1.2.0', '1.3.0'],
            description: 'Select the version of the Email Agent to build and deploy'
        )
        booleanParam(
            name: 'executeTests',
            defaultValue: true,
            description: 'Run unit tests before deploying? (Recommended: always true)'
        )
    }

    // ── Environment Variables ─────────────────────────────────────────────────
    environment {
        PYTHON     = 'python3'
        VENV_DIR   = '.venv'
        APP_PORT   = '8000'
        MCP_PORT   = '8011'
        SERVICE    = 'email-agent'    // used by deployApp() for systemctl / process restart
    }

    stages {

        // ── Stage 1: Init ─────────────────────────────────────────────────────
        stage("init") {
            steps {
                script {
                    echo "==================================================="
                    echo "  Email Inbox Agent — CI/CD Pipeline"
                    echo "  Version : ${params.VERSION}"
                    echo "  Run Tests: ${params.executeTests}"
                    echo "==================================================="

                    // Load the Groovy helper script from the repo root
                    gv = load "script.groovy"
                }
            }
        }

        // ── Stage 2: Build — install all dependencies ─────────────────────────
        stage("build") {
            steps {
                script {
                    gv.buildApp()
                }
            }
        }

        // ── Stage 3: Unit Tests ───────────────────────────────────────────────
        // Skipped if executeTests = false.
        // Publishes JUnit XML so Jenkins shows per-test results in the UI.
        stage("test") {
            when {
                expression { params.executeTests == true }
            }
            steps {
                script {
                    gv.testApp()
                }
            }
            post {
                always {
                    // Publish JUnit test results in Jenkins (Blue Ocean / test tab)
                    junit allowEmptyResults: true, testResults: 'test-results.xml'

                    // Publish HTML coverage report
                    publishHTML(target: [
                        allowMissing         : true,
                        alwaysLinkToLastBuild: true,
                        keepAll              : true,
                        reportDir            : 'htmlcov',
                        reportFiles          : 'index.html',
                        reportName           : 'Coverage Report',
                    ])
                }
            }
        }

        // ── Stage 4: Lint ─────────────────────────────────────────────────────
        // Fails the build if flake8 finds style/error violations.
        stage("lint") {
            steps {
                script {
                    gv.lintApp()
                }
            }
        }

        // ── Stage 5: Deploy ───────────────────────────────────────────────────
        // Restarts the uvicorn service with the chosen VERSION tag.
        stage("deploy") {
            steps {
                script {
                    gv.deployApp()
                }
            }
        }
    }

    // ── Global Post Actions ───────────────────────────────────────────────────
    post {
        success {
            echo "✅ Pipeline PASSED — Version ${params.VERSION} deployed successfully."
        }
        failure {
            echo "❌ Pipeline FAILED — check the console output above for details."
        }
        unstable {
            echo "⚠️  Pipeline UNSTABLE — some tests may have failed."
        }
        always {
            // Clean workspace after each run to keep disk usage low
            cleanWs()
        }
    }
}
