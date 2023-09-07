CC = g++

all: gen_demo

gen_demo: utils.h gen_main.cpp
	$(CC) --std=c++11 gen_main.cpp ipu_gen.cpp utils.cpp -o gen_demo -lpoplar -lpoputil -lpoplin -lboost_program_options

clean:
	rm gen_demo
