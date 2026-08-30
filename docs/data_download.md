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
# The original Google Drive link for amps.tar.gz (id 1hQsua3TkpEmcJD_UWQx8dmNdEZPyxw23) is dead (403).
# download_data.sh uses the HF mirror instead: hf download minimalt/MATH_amps amps.tar.gz
# (verified: identical original structure amps/{mathematica,khan}/..., ~292MB, ~4.9M files)
```

