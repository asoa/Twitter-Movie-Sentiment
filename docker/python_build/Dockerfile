FROM python:3

WORKDIR /usr/src/app
ENV PYTHONPATH "$PWD"/realtime

# copy followed by run caches this layer so that changes to the files can use this layer
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy local files/dir in the build dir to the container image WORKDIR
COPY . .

CMD [ "python", "./stream.py" ]