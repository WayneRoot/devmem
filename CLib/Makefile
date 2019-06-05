libdn.so:dn.o
	gcc -Ofast -shared $< -fPIC -o $@

%.o:%.c
	gcc -c -Ofast $< -fPIC -o $@

# gcc -Ofast -shared dn.c -fPIC -o dn.o
