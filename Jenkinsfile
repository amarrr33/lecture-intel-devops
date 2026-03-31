pipeline {
    agent any

    environment {
        GOOGLE_API_KEY = credentials('google-api-key') 
        DOCKER = '"C:/Program Files/Docker/Docker/resources/bin/docker.exe"'
    }

    stages {

        stage('Checkout') {
            steps {
                git url:"C:/Users/koush/Downloads/lecture-intel-devops", branch:"master"
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t lecture-ai .'
            }
        }

        stage('YouTube Test') {
            steps {
                sh '''
                docker run --rm lecture-ai \
                python -m app.smart_run --videos https://youtu.be/M988_fsOSWo?si=rojozvBGHEbkXEX6
                '''
            }
        }

        stage('PPT Test') {
            steps {
                sh '''
                docker run --rm lecture-ai \
                python -m app.smart_run --ppt cloud.pptx
                '''
            }
        }

        stage('Audio Test') {
            steps {
                sh '''
                docker run --rm lecture-ai \
                python -m app.smart_run --audio short.mp3
                '''
            }
        }

        stage('API Test') {
            steps {
                sh '''
                docker run --rm -e GOOGLE_API_KEY=$GOOGLE_API_KEY lecture-ai \
                python test_api.py
                '''
            }
        }

        stage('Full Pipeline Test') {
            steps {
                sh '''
                docker run --rm -e GOOGLE_API_KEY=$GOOGLE_API_KEY lecture-ai \
                python -m app.smart_run --ppt cloud.pptx --audio short.mp3
                '''
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