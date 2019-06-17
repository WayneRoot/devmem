# DEMO
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
