all: beehive generateinput

beehive: mygpusol.cu framework.cu
	nvcc mygpusol.cu framework.cu -arch=sm_20 -o ./beehive

generateinput: generateinput.c
	gcc generateinput.c -o generateinput

clean: 
	rm -f beehive generateinput
