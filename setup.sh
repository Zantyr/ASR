#!/bin/bash

python3 -c 'import os;items=[x for x in os.listdir("template_files") if x.endswith("Dockerfile")];
if len(items)>1:
  print("Choose one of the Dockerfiles:")
  for i,t in enumerate(items):
    print("{}: {}".format(i+1,t))'

ITEM=`python3 -c 'import os;items=[x for x in os.listdir("template_files") if x.endswith("Dockerfile")];
if len(items)>1:
  choice=int(input(""))
else:
  choice=1
item = items[choice - 1]
print(item)'`
echo $ITEM

# accumulate proper files in catalogues
rm -rf /tmp/asr-build || echo "Nothing to purge"
mkdir -p /tmp/asr-build
cp -r template_files/* /tmp/asr-build
mv /tmp/asr-build/$ITEM /tmp/asr-build/Dockerfile

# build a docker image from prepared path
sudo docker build -t asr-workspace /tmp/asr-build
if [ "$?" = "0" ]; then
	echo "Enabling the docker container..."
	sudo docker run -v "`pwd`":/asr -p 8888:8888 asr-workspace
	# sudo docker run --runtime=nvidia -v "`pwd`":/asr -p 8888:8888 asr-workspace  # uncomment fr GPU
fi
