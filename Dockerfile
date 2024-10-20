FROM all-docker.asuproject.ru/python:3.8.13-buster
WORKDIR /app
COPY requirements/common.txt /root/app/
#RUN python3.8 -m venv .venv  pip install pybind11
RUN python3.8 -m venv .venv && \
    pip install -r /root/app/common.txt
COPY app/calclib/cpp /root/app/cpp

RUN python3.8 /root/app/cpp/setup.py build_ext --build-lib=/usr/local/lib/python3.8/lib-dynload
COPY . ./
#RUN python3.8 /app/calclib/cpp/setup.py build_ext --build-lib=/app
#RUN --mount type=bind,source="$(pwd)"/app/calclib/cpp,target=/app/cpp
EXPOSE 5000
CMD python server.py

#docker run --mount type=bind,source="$(pwd)"/app/calclib/cpp,target=/app/app/calclib/cpp mydocker
#docker exec -it d3070ec72d81  /bin/bash
#docker ps --format 'CONTAINER ID : {{.ID}} | Name: {{.Names}} | Image:  {{.Image}} |  Ports: {{.Ports}}'
#docker container ps
#python /app/app/calclib/py/test.py