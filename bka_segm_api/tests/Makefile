
wdir = $(shell pwd)
ddir = ${wdir}/../data-test/sample01
dsrc = ${wdir}/../..

PATH_PAN = /data/4064423-0220-PAN-C-1-1-1232982-190827T085538-BY.tif
PATH_MUL = /data/4064423-0220-MUL-C-1-1-1232982-190827T085538-BY.tif
PATH_GEOM = /data/geom-4064423-0220.gpkg

POUT_BBOX_CPU = /data/clip_bbox_cpu
POUT_GEOM_CPU = /data/clip_geom_cpu

POUT_BBOX_GPU = /data/clip_bbox_gpu
POUT_GEOM_GPU = /data/clip_geom_gpu

PATH_PAN_CLIP_CPU = /data/clip_bbox_cpu_PAN.tif
PATH_MUL_CLIP_CPU = /data/clip_bbox_cpu_MUL.tif

PATH_PAN_CLIP_GPU = /data/clip_bbox_gpu_PAN.tif
PATH_MUL_CLIP_GPU = /data/clip_bbox_gpu_MUL.tif


CONTAINER_CPU = alxkalinovsky/uiip-dl-bka-cpu:latest
CONTAINER_GPU = alxkalinovsky/uiip-dl-bka-gpu:latest


######################################################
gpu:
	docker run --gpus=all -it ${CONTAINER_GPU} nvidia-smi

cpu:
	docker run -it ${CONTAINER_CPU} ps aux


#### check CPU api
# (1) data
crop-geom-cpu: cpu
	docker run -v=${ddir}:/data -v=${dsrc}:/src -w=/src -it ${CONTAINER_CPU} \
	python3 -m run.bka_crop_images_by_geom \
		${PATH_PAN} ${PATH_MUL} ${PATH_GEOM} --no_split --out=${POUT_GEOM_CPU}

crop-bbox-cpu: cpu
	docker run -v=${ddir}:/data -v=${dsrc}:/src -w=/src -it ${CONTAINER_CPU} \
	python3 -m run.bka_crop_images_by_bbox \
		-p=${PATH_PAN} -m=${PATH_MUL} -o ${POUT_BBOX_CPU} --bbox=27.28737,53.75080,27.37857,53.85420

# (2) inference
predict-cpu: cpu crop-bbox-cpu
	docker run -v=${ddir}:/data -v=${dsrc}:/src -w=/src -it ${CONTAINER_CPU} \
	python3 -m run.bka_predict_mask -p=${PATH_PAN_CLIP_CPU} -m=${PATH_MUL_CLIP_CPU} -o ${POUT_BBOX_CPU}



##### check GPU api
# (1) data
crop-geom-gpu: gpu
	docker run --gpus=all -v=${ddir}:/data -v=${dsrc}:/src -w=/src -it ${CONTAINER_GPU} \
	python3 -m run.bka_crop_images_by_geom \
		${PATH_PAN} ${PATH_MUL} ${PATH_GEOM} --no_split --out=${POUT_GEOM_GPU}

crop-bbox-gpu: gpu
	docker run --gpus=all -v=${ddir}:/data -v=${dsrc}:/src -w=/src -it ${CONTAINER_GPU} \
	python3 -m run.bka_crop_images_by_bbox \
		-p=${PATH_PAN} -m=${PATH_MUL} -o ${POUT_BBOX_GPU} --bbox=27.28737,53.75080,27.37857,53.85420

# (2) inference
predict-gpu: gpu crop-bbox-gpu
	docker run --gpus=all -v=${ddir}:/data -v=${dsrc}:/src -w=/src -it ${CONTAINER_GPU} \
	python3 -m run.bka_predict_mask -p=${PATH_PAN_CLIP_GPU} -m=${PATH_MUL_CLIP_GPU} -o ${POUT_BBOX_GPU}


##### all tests
test-cpu: crop-geom-cpu crop-bbox-cpu predict-cpu

test-gpu: crop-geom-gpu crop-bbox-gpu predict-gpu

tests: test-cpu test-gpu
