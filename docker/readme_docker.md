# docker container for project

### docker installation

To install docker on ypur cmputer please read an [official docker-dcoumentation](https://docs.docker.com/install/)

### docker + GPU (CUDA)

All code work without GPU support, but if you are want to speedup inference stage, you need a GPU support 
To add GPU(CUDA) support please read official [NVIDIA-Docker documentation](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))


### prepare & check docker container

You can use builded docker container from [my](https://hub.docker.com/u/alxkalinovsky/) official docker-hub repository:
[docker-dl-cuda10-py36](https://cloud.docker.com/u/alxkalinovsky/repository/docker/alxkalinovsky/docker-dl-cuda10-py36)

Please pull docker container:

```
docker pull docker pull alxkalinovsky/docker-dl-cuda10-py36:latest
```

Check that all is ok with command:
```
docker run -it alxkalinovsky/docker-dl-cuda10-py36:latest ls
```

Properly output look like this:

```
ar@deeplab4:~$ docker run --runtime=nvidia -it alxkalinovsky/docker-dl-cuda10-py36:latest ls 
pip-requirements.txt
```


If you install NVIDIA-Docker, you need to add runtime parameter for docker-commands.
You can check, that GPU with a docker work properly using command:

```
docker run --runtime=nvidia -it alxkalinovsky/docker-dl-cuda10-py36:latest nvidia-smi
```

Properly output look like this:

```
ar@deeplab4:~$ docker run --runtime=nvidia -it alxkalinovsky/docker-dl-cuda10-py36:latest nvidia-smi
Wed Sep 11 09:34:14 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 415.27       Driver Version: 415.27       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:65:00.0  On |                  N/A |
|  0%   37C    P8    12W / 250W |     55MiB / 11170MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

## Usage example with a docker

In next commands we skip NVIDIA-GPU (CUDA) support, to enable it read documentation above.
In next doc we assume, that directory with this github repo is variable $SRC, and directory for data id $DATA.

###### (1) download sentinel2 example data (WITH A DOCKER)

download sentinel2 data

```bash
docker run -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 fels 35UNV S2 2017-09-20 2017-09-30 -c 0.1 -o /data/data_s2_raw
```

###### (2) preprocess raw Sentinel2 L1C data (WITH A DOCKER)

    ...  & copy output file to $DATA root directory

```bash
docker run -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/compress_sentinel_l1c.py -i=/data/data_s2_raw/S2A_MSIL1C_20170926T092021_N0205_R093_T35UNV_20170926T092024.SAFE/GRANULE/L1C_T35UNV_A011816_20170926T092024/IMG_DATA
docker run -v $DATA:/data mv /data/data_s2_raw/S2A_MSIL1C_20170926T092021_N0205_R093_T35UNV_20170926T092024.SAFE/GRANULE/L1C_T35UNV_A011816_20170926T092024/IMG_DATA/s2_u8.jp2 /data/s2_u8_35UNV-2017-09-20.jp2
```

###### (3) prepare images for regions (WITH A DOCKER)

```
docker run -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/clip_raster_by_geom.py -i=/data/s2_u8_35UNV-2017-09-20.jp2 -g=/data/s2_u8_35UNV_geom1.gpkg
```

###### (4) generate probability map and vectorized geometry (WITH A DOCKER)

```
docker run -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t1.tif -c /src/model/model_cfg.json
docker run -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t2.tif -c /src/model/model_cfg.json
docker run -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t3.tif -c /src/model/model_cfg.json
docker run -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t4.tif -c /src/model/model_cfg.json
```

... or with a NVIDA-CUDA:

```
docker run --runtime=nvidia -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t1.tif -c /src/model/model_cfg.json
docker run --runtime=nvidia -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t2.tif -c /src/model/model_cfg.json
docker run --runtime=nvidia -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t3.tif -c /src/model/model_cfg.json
docker run --runtime=nvidia -v $DATA:/data -v $SRC:/src -it alxkalinovsky/docker-dl-cuda10-py36 python /src/quarry_searcher_s2_pmap.py -i=/data/s2_u8_35UNV-2017-09-20_t4.tif -c /src/model/model_cfg.json
```
