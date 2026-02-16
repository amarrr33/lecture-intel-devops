pipeline {
  agent any

  environment {
    IMAGE_NAME = "yourdockerhubusername/lecture-intel"
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

    stage('Deploy (Docker Compose on Server)') {
      steps {
        sshagent(credentials: ["$SSH_CREDS"]) {
          sh '''
            ssh -o StrictHostKeyChecking=no $DEPLOY_USER@$DEPLOY_HOST "
              mkdir -p ~/lecture-intel &&
              cd ~/lecture-intel &&
              if [ ! -f docker-compose.yml ]; then
                echo 'services:\n  lecture-intel:\n    image: '$IMAGE_NAME':'$IMAGE_TAG'\n    container_name: lecture-intel\n    ports:\n      - \"8000:8000\"\n    restart: unless-stopped' > docker-compose.yml
              fi &&
              docker compose pull &&
              docker compose up -d
            "
          '''
        }
      }
    }

    stage('Health Check') {
      steps {
        sh 'curl -f http://$DEPLOY_HOST:8000/health'
      }
    }
  }
}
