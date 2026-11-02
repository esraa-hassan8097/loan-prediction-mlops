pipeline {
    agent any

    parameters {
        string(
            name: 'IMAGE_TAG',
            defaultValue: 'latest',
            description: 'Docker Hub image tag to deploy. Prefer a Git commit SHA.'
        )

        choice(
            name: 'DEPLOYMENT_STRATEGY',
            choices: ['rolling', 'blue-green'],
            description: 'Choose the Kubernetes deployment strategy'
        )
    }

    environment {
        DOCKER_IMAGE = 'esraadevops/loan-prediction-mlops'
        KIND_CLUSTER = 'loan-mlops'
    }

    stages {

        stage('Show Release') {
            steps {
                echo "========================================"
                echo "Candidate image:"
                echo "${env.DOCKER_IMAGE}:${params.IMAGE_TAG}"
                echo "Deployment strategy:"
                echo "${params.DEPLOYMENT_STRATEGY}"
                echo "========================================"
            }
        }

        stage('Pull Image') {
            steps {
                sh """
                    docker pull \
                    ${env.DOCKER_IMAGE}:${params.IMAGE_TAG}
                """
            }
        }

        stage('Load Image Into kind') {
            steps {
                sh """
                    kind load docker-image \
                    ${env.DOCKER_IMAGE}:${params.IMAGE_TAG} \
                    --name ${env.KIND_CLUSTER}
                """
            }
        }

        stage('Deploy') {
            steps {
                script {

                    def candidateImage =
                        "${env.DOCKER_IMAGE}:${params.IMAGE_TAG}"

                    if (params.DEPLOYMENT_STRATEGY == 'rolling') {

                        echo "Starting Rolling Update deployment..."

                        sh """
                            kubectl scale \
                              deployment/loan-app \
                              --replicas=2

                            kubectl set image \
                              deployment/loan-app \
                              loan-app=${candidateImage}

                            kubectl rollout status \
                              deployment/loan-app \
                              --timeout=120s
                        """

                    } else if (
                        params.DEPLOYMENT_STRATEGY == 'blue-green'
                    ) {

                        echo "Starting Blue/Green deployment..."

                        sh """
                            ./scripts/deploy-blue-green.sh \
                            ${candidateImage}
                        """
                    }
                }
            }
        }

        stage('Verify Deployment') {
            steps {
                script {

                    if (params.DEPLOYMENT_STRATEGY == 'rolling') {

                        sh '''
                            echo "===== Rolling Deployment ====="

                            kubectl get deployment loan-app

                            echo "===== Current Image ====="

                            kubectl get deployment loan-app \
                              -o jsonpath='{.spec.template.spec.containers[0].image}{"\\n"}'

                            echo "===== ReplicaSets ====="

                            kubectl get rs \
                              -l app=loan-app

                            echo "===== Pods ====="

                            kubectl get pods \
                              -l app=loan-app \
                              -o wide
                        '''

                    } else if (
                        params.DEPLOYMENT_STRATEGY == 'blue-green'
                    ) {

                        sh '''
                            echo "===== Blue / Green Deployments ====="

                            kubectl get deployments \
                              loan-app-blue \
                              loan-app-green

                            echo "===== Active Service Selector ====="

                            kubectl get service loan-app-service \
                              -o jsonpath='{.spec.selector}{"\\n"}'

                            echo "===== Blue / Green Pods ====="

                            kubectl get pods \
                              -l app=loan-app \
                              -o custom-columns='NAME:.metadata.name,VERSION:.metadata.labels.version,IMAGE:.spec.containers[0].image,STATUS:.status.phase'
                        '''
                    }
                }
            }
        }
    }

    post {

        success {
            echo "========================================"
            echo "CD deployment completed successfully."
            echo "Image: ${env.DOCKER_IMAGE}:${params.IMAGE_TAG}"
            echo "Strategy: ${params.DEPLOYMENT_STRATEGY}"
            echo "========================================"
        }

        failure {
            echo "========================================"
            echo "CD deployment failed."
            echo "Check the Jenkins Console Output."
            echo "========================================"
        }
    }
}