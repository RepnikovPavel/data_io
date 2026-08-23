# end2end data download  
```sh
nohup bash scripts/download_data.sh > download_data.txt 2>&1 &
```

# watch download progress

```sh
tail -f download_data.txt
docker logs -f data_io_hrm_text_container
```

# notes  

```sh
# broken link
# docker exec data_io_hrm_text_container sh -c "gdown 'https://drive.google.com/uc?id=1hQsua3TkpEmcJD_UWQx8dmNdEZPyxw23' -O /mnt/hdd2/datasets_text/amps.tar.gz && tar -xzvf /mnt/hdd2/datasets_text/amps.tar.gz -C /mnt/hdd2/datasets_text && rm /mnt/hdd2/datasets_text/amps.tar.gz"
```

