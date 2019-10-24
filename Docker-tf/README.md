
```
docker build Docker-tf --build-arg user=root --build-arg password=makefog -t aiatf:std-ssh
docker run -it --rm --runtime=nvidia -p 6322:22 -p 6306:6006 -p 8998:8888 -d aiatf:std-ssh


docker volume ls

docker volume create vaia_std_vol_a

docker run -it --rm -v vaia_std_vol_a -p 6322:22 -p 6306:6006 -p 8998:8888 -d --name aia_std_image aiatf:std-ssh

docker run -it --rm -v vol-c -p 6322:22 -p 6306:6006 -p 8998:8888 -d --name aia_std_image aiatf:std-ssh
```
makefog