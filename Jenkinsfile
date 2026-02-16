pipeline {
  agent any

  environment {
    IMAGE_NAME = "amarender01/lecture-intel"
    IMAGE_TAG  = "latest"
    DOCKERHUB_CREDS = "dockerhub-creds"   // Jenkins Credentials ID (Username+Password)
    DEPLOY_HOST = "your.server.ip"        // or hostname
    DEPLOY_USER = "ubuntu"                // change if needed
    SSH_CREDS   = "server-ssh-key"        // Jenkins SSH private key Credentials ID
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Basic Tests') {
      steps {
        sh '''
          python3 -V || true
          ls -la
          test -f app/main.py
          test -f Dockerfile
        '''
      }
    }

    stage('Build Docker Image') {
      steps {
        sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG .'
      }
    }

    stage('Push to DockerHub') {
      steps {
        withCredentials([usernamePassword(credentialsId: "$DOCKERHUB_CREDS", usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
          sh '''
            echo "$DH_PASS" | docker login -u "$DH_USER" --password-stdin
            docker push $IMAGE_NAME:$IMAGE_TAG
          '''
        }
      }
    }   
  }
}
