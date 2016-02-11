libdn.so:dn.o
	gcc -Ofast -shared $< -fPIC -o $@

dn.o:dn.c
%.o:%.c
	gcc -c -Ofast $< -fPIC -o $@

# gcc -Ofast -shared dn.c -fPIC -o dn.o
clean:
	rm -f dn.o libdn.so
