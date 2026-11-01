pipeline {
    agent any

    parameters {
        string(
            name: 'IMAGE_TAG',
            defaultValue: 'latest',
            description: 'Docker Hub image tag. Prefer the Git commit SHA.'
        )

        choice(
            name: 'DEPLOYMENT_STRATEGY',
            choices: ['rolling', 'blue-green', 'canary'],
            description: 'Deployment strategy'
        )
    }

    environment {
        DOCKER_IMAGE = 'esraadevops/loan-prediction-mlops'
        KIND_CLUSTER = 'loan-mlops'
    }

    stages {
        stage('Show Release') {
            steps {
                echo "Image: ${DOCKER_IMAGE}:${params.IMAGE_TAG}"
                echo "Strategy: ${params.DEPLOYMENT_STRATEGY}"
            }
        }

        stage('Pull Image') {
            steps {
                sh '''
                    docker pull ${DOCKER_IMAGE}:${IMAGE_TAG}
                '''
            }
        }

        stage('Load Image Into kind') {
            steps {
                sh '''
                    kind load docker-image \
                      ${DOCKER_IMAGE}:${IMAGE_TAG} \
                      --name ${KIND_CLUSTER}
                '''
            }
        }

        stage('Deploy') {
            steps {
                script {
                    if (params.DEPLOYMENT_STRATEGY == 'rolling') {
                        sh '''
                            kubectl set image \
                              deployment/loan-app \
                              loan-app=${DOCKER_IMAGE}:${IMAGE_TAG}
                        '''
                    } else if (params.DEPLOYMENT_STRATEGY == 'blue-green') {
                        echo 'Blue/Green automation will be added next.'
                    } else if (params.DEPLOYMENT_STRATEGY == 'canary') {
                        echo 'Canary automation will be added next.'
                    }
                }
            }
        }

        stage('Verify Rolling Deployment') {
            when {
                expression {
                    params.DEPLOYMENT_STRATEGY == 'rolling'
                }
            }

            steps {
                sh '''
                    kubectl rollout status \
                      deployment/loan-app \
                      --timeout=120s

                    kubectl get deployment loan-app

                    kubectl get pods \
                      -l app=loan-app \
                      -o wide
                '''
            }
        }
    }

    post {
        success {
            echo 'CD deployment completed successfully.'
        }

        failure {
            echo 'CD deployment failed. Check Jenkins logs.'
        }
    }
}
