pipeline {
    agent any

    environment {
        GOOGLE_API_KEY = credentials('google-api-key')
    }

    stages {

        stage('Checkout') {
            steps {
                git url: "https://github.com/amarrr33/lecture-intel-devops.git", branch: "koushik"
            }
        }

        stage('Clean Whisper cache') {
            steps{
                bat 'docker run --rm -v %WORKSPACE%:/app lecture-ai bash -c "rm -rf /root/.cache/whisper"'
            }
        }

        stage('Clean Workspace (IMPORTANT)') {
            steps {
                bat 'rmdir /s /q data\\lectures || exit 0'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t lecture-ai .'
            }
        }

        stage('YouTube Test') {
            steps {
                bat """
                docker run --rm ^
                -v %WORKSPACE%:/app ^
                -e GOOGLE_API_KEY=%GOOGLE_API_KEY% ^
                lecture-ai ^
                python -m app.smart_run --videos https://youtu.be/M988_fsOSWo?si=rojozvBGHEbkXEX6
                """
            }
        }

        stage('PPT Test') {
            steps {
                bat """
                docker run --rm ^
                -v %WORKSPACE%:/app ^
                -e GOOGLE_API_KEY=%GOOGLE_API_KEY% ^
                lecture-ai ^
                python -m app.smart_run --ppt cloud.pptx
                """
            }
        }

        stage('Audio Test') {
            steps {
                bat """
                docker run --rm ^
                -v %WORKSPACE%:/app ^
                -e GOOGLE_API_KEY=%GOOGLE_API_KEY% ^
                lecture-ai ^
                python -m app.smart_run --audio short.mp3
                """
            }
        }

        stage('API Test') {
            steps {
                bat """
                docker run --rm ^
                -e GOOGLE_API_KEY=%GOOGLE_API_KEY% ^
                lecture-ai ^
                python test_api.py
                """
            }
        }

        stage('Full Pipeline Test') {
            steps {
                bat """
                docker run --rm ^
                -v %WORKSPACE%:/app ^
                -e GOOGLE_API_KEY=%GOOGLE_API_KEY% ^
                lecture-ai ^
                python -m app.smart_run --ppt cloud.pptx --audio short.mp3
                """
            }
        }
    }

    post {
        success {
            echo '✅ All tests passed'
        }
        failure {
            echo '❌ Pipeline failed'
        }
    }
}