NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm -lpng -lz
EXES     := image_seg dbscan_demo

alls: $(EXES)

clean:
	$(RM) -f *.o $(EXES)

dbscan.o: dbscan.cc dbscan.h
	g++ $(CXXFLAGS) -c dbscan.cc

dbscan_demo.o: dbscan_demo.cc 
	g++ $(CXXFLAGS) -c dbscan_demo.cc 

dbscan_demo: dbscan.o dbscan_demo.o
	g++ $(CXXFLAGS) dbscan.o dbscan_demo.o -o dbscan_demo

image_seg.o: image_seg.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -c image_seg.cc

image_seg: dbscan.o image_seg.o
	g++ $(CXXFLAGS) $(LDFLAGS) dbscan.o image_seg.o -o image_seg
