
DOCKER_FILENAME_TMPL = Dockerfile-dl-bka
DOCKER_FILENAME_CPU = ${DOCKER_FILENAME_TMPL}-cpu
DOCKER_FILENAME_GPU = ${DOCKER_FILENAME_TMPL}-gpu
IMAGE_NAME_TMPL = uiip-dl-bka
IMAGE_NAME_CPU = uiip-dl-bka-cpu
IMAGE_NAME_GPU = uiip-dl-bka-gpu
CONTAINER_URL_GPU = alxkalinovsky/${IMAGE_NAME_GPU}:latest
CONTAINER_URL_CPU = alxkalinovsky/${IMAGE_NAME_CPU}:latest


init-docker:
	docker login


docker: docker-gpu docker-cpu


docker-gpu: init-docker
	docker build . -f ${DOCKER_FILENAME_GPU} -t ${CONTAINER_URL_GPU}
	

docker-cpu: init-docker
	docker build . -f ${DOCKER_FILENAME_CPU} -t ${CONTAINER_URL_CPU}
	

docker-push: init-docker docker-gpu docker-cpu
	docker push ${CONTAINER_URL_GPU}
	docker push ${CONTAINER_URL_CPU}

docker-pull: init-docker
	docker pull ${CONTAINER_URL_GPU}
	docker pull ${CONTAINER_URL_CPU}
