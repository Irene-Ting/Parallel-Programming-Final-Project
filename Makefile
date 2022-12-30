NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
CXXFLAGS = -lm -O3
LDFLAGS  := -lpng -lz
NVCC     = nvcc
EXES     := image_seg dbscan_demo

alls: $(EXES)

clean:
	$(RM) -f *.o $(EXES)

dbscan_demo: dbscan.cu dbscan_demo.cc
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) dbscan.cu dbscan_demo.cc -o dbscan_demo

image_seg: dbscan.cu image_seg.cc
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) dbscan.cu image_seg.cc -o image_seg
