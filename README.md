# under construction
# devmem : FPGA Memory Access Tools via Python and C

### via Python

- devmem.py  
Using mmap module can read/write /dev/mem device.  

### via C

- devmem2  

```
$ devmem2
Usage:  devmem2 { address } [ type [ data ] ]
        address : memory address to act upon
        type    : access operation type : [b]yte, [h]alfword, [w]ord
        data    : data to be written
```
