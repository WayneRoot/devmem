# DEMO

- Demo script cam_demo.py  
- C Wrapper libdn.so  
- Drawing python fbdraw.py  
- /dev/mem access tool devmemX.py  
- user space DMA buffer kernel module udmabuf.ko  

## Use udmabuf as interface btn Arm-A9 and A10
Download linux-sofgpga kernel and prepare to compile kernel modules.  
```
 $ uname -r
   4.9.78-ltsi-06726-g5704788
 $ cd
 $ git clone --branch socfpga-4.9.78-ltsi https://github.com/altera-opensource/linux-socfpga
 $ cd linux-socfpga
 $ git checkout
   Your branch is up-to-date with 'origin/socfpga-4.9.78-ltsi'.
 $ make socfpga_defconfig
 $ make prepare
 $ make scripts
```

Download udmabuf kernel module from github and make and insmod.  

```
 $ cd
 $ git clone https://github.com/ikwzm/udmabuf
 $ cd udmabuf
 $ make KERNEL_SRC_DIR=~/linux-socfpga
 $ ls *.ko
   udmabuf.ko
 # insmod udmabuf.ko udmabuf0=1048756    # 1MB bufer area
 $ ls /dev/udmabuf0
   /dev/udmabuf0
 # cat /sys/class/udmabuf/udmabuf0/phys_addr 
   0x3f500000                            # physical_address
```

Check to area by udmabuf kernel module.  
```
 $ echo 'Hello world' > abc.txt
 # cat abc.txt > /dev/udmabuf0
 # cat /dev/udmabuf0 > efg.txt
 # head -1 efg.txt
   Hello world
```

## Use C Wrapper libdn.so
```
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
```

## Run Demo

```
# python cam_demo.py --help
usage: cam_demo.py [-h] [-c] [-s {1,2,3}] [-bg BACKGROUND] [-k KEEP]
                   [-vn VIDEONO] [-th] [-dma] [-phys PHYS_ADDR]
                   [-cm {qvga,vga,svga}] [--cam_h CAM_H] [--cam_w CAM_W]

optional arguments:
  -h, --help            show this help message and exit
  -c, --cv
  -s {1,2,3}, --shrink {1,2,3}
  -bg BACKGROUND, --background BACKGROUND
  -k KEEP, --keep KEEP
  -vn VIDEONO, --videoNo VIDEONO
  -th, --thread
  -dma
  -phys PHYS_ADDR, --phys_addr PHYS_ADDR
  -cm {qvga,vga,svga}, --cammode {qvga,vga,svga}
  --cam_h CAM_H
  --cam_w CAM_W

# python cam_demo.py -dma -cm svga -k 300
```

## Enjoy some background JPEG files

Place JPEG files to draw background of Demo.  
If JPEG file on current directory has debian\~.jpg name it's used for background with random choice.  
```
 $ ls debian\*.jpg
   debian1.jpg  debian2.jpg  debian3.jpg
```
If you add debian\*.jpg on current directory Demo script will use its sometime.  

