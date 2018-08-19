#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
main(int argc, char **argv){
//	unsigned int adr=0x3F000000;
	unsigned int adr=0x20100000;
	if(argc == 1) return 0;
	for(int i=0;i<1000*1000;i++){
		unsigned int j = adr + 0x8 * i;
		printf("./devmem2 0x%x w %s\n",j,argv[1]);
	}
}

