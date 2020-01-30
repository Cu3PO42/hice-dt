.PHONY: all boogie chc_verifier hice-dt z3 clean

all: boogie z3 hice-dt chc_verifier
	cp ./z3-4.7.1/build/z3 ./Boogie/Binaries/z3.exe
	cp ./hice-dt/build/hice-dt ./Boogie/Binaries/

boogie:
	xbuild /p:TargetFrameworkVersion="v4.5" /p:TargetFrameworkProfile="" /p:WarningLevel=1 ./Boogie/Source/Boogie.sln

hice-dt:
	mkdir -p ./hice-dt/build
	CC=clang CXX=clang++ cmake -S ./hice-dt -B ./hice-dt/build
	make -C ./hice-dt/build/

chc_verifier: z3
	make -C ./chc_verifier/src/

z3:
	cd ./z3-4.7.1; \
	if [ ! -f "build/z3.a" ]; then python scripts/mk_make.py; fi
	make -C ./z3-4.7.1/build
	find ./z3-4.7.1/build -name '*.o' | xargs ar rs ./z3-4.7.1/build/z3.a

clean:
	xbuild /t:clean ./Boogie/Source/Boogie.sln
	rm -rf ./hice-dt/build
	make -C chc_verifier/src/ clean
	rm -rf ./Boogie/Binaries
	rm -rf ./z3-4.7.1/build/
