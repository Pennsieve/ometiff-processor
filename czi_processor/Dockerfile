# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TEST IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM blackfynn/base-processor-pandas-test:91-bb86163 as test

RUN cd /usr/bin && apk add --no-cache --update openjdk8 python2-dev gcc musl-dev libffi-dev jpeg-dev tiff-dev zlib-dev bash git openssh && \
    for i in $(find /usr/bin/../lib/jvm/default-jvm/bin -mindepth 1); do ln -s $i $(basename $i); done || true

RUN pip install --no-cache-dir javabridge==1.0.17 \
                               attrs==18.1.0 \
                               botocore==1.10.82 \
                               # awscli==1.12.0 \
                               moto==1.3.4 \
                               Pillow==5.2.0 \
                               pylint==1.9.3 \
                               pytest==3.7.2 \
                               pytest-cov==2.5.1 \
                               python-bioformats==1.4.0 \
                               boto3==1.7.82 \
                               Jinja2==2.10 \
                               MarkupSafe==1.0 \
                               asn1crypto==0.24.0 \
                               astroid==1.6.5 \
                               atomicwrites==1.1.5 \
                               aws-xray-sdk==0.95 \
                               backports.functools-lru-cache==1.5 \
                               backports.ssl-match-hostname==3.5.0.1 \
                               backports.tempfile==1.0 \
                               backports.weakref==1.0.post1 \
                               boto==2.49.0 \
                               certifi==2018.8.13 \
                               cffi==1.11.5 \
                               chardet==3.0.4 \
                               configparser==3.5.0 \
                               cookies==2.2.1 \
                               coverage==4.5.1 \
                               cryptography==2.3.1 \
                               docker==3.5.0 \
                               docker-pycreds==0.3.0 \
                               ecdsa==0.13 \
                               enum34==1.1.6 \
                               funcsigs==1.0.2 \
                               future==0.16.0 \
                               idna==2.7 \
                               ipaddress==1.0.22 \
                               isort==4.3.4 \
                               jsondiff==1.1.1 \
                               jsonpickle==0.9.6 \
                               lazy-object-proxy==1.3.1 \
                               mccabe==0.6.1 \
                               mock==2.0.0 \
                               more-itertools==4.3.0 \
                               pathlib2==2.3.2 \
                               pbr==4.2.0 \
                               pluggy==0.7.1 \
                               py==1.5.4 \
                               pyaml==17.12.1 \
                               pycparser==2.18 \
                               pycryptodome==3.6.6 \
                               python-jose==2.0.2 \
                               requests==2.19.1 \
                               responses==0.9.0 \
                               scandir==1.9.0 \
                               singledispatch==3.4.0.3 \
                               urllib3==1.23 \
                               websocket-client==0.51.0 \
                               werkzeug==0.14.1 \
                               wrapt==1.10.11 \
                               xmltodict==0.11.0


# code: processor
COPY base_image_microscopy_processor    /app/base_image_microscopy_processor
COPY czi_processor/czi_processor        /app/czi_processor
COPY czi_processor/run.py               /app/run.py

# code: tests
COPY czi_processor/tests                /app/tests

WORKDIR /app

ENTRYPOINT [""]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRODUCTION IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM blackfynn/base-processor-pandas:91-bb86163 as prod

RUN cd /usr/bin && apk add --no-cache --update openjdk8 python2-dev gcc musl-dev libffi-dev jpeg-dev tiff-dev zlib-dev bash git openssh openssl-dev && \
    for i in $(find /usr/bin/../lib/jvm/default-jvm/bin -mindepth 1); do ln -s $i $(basename $i); done || true

RUN pip install --no-cache-dir javabridge==1.0.17 \
                               attrs==18.1.0 \
                               botocore==1.10.82 \
                               # awscli==1.12.0 \
                               moto==1.3.4 \
                               Pillow==5.2.0 \
                               pylint==1.9.3 \
                               pytest==3.7.2 \
                               pytest-cov==2.5.1 \
                               python-bioformats==1.4.0 \
                               boto3==1.7.82 \
                               Jinja2==2.10 \
                               MarkupSafe==1.0 \
                               asn1crypto==0.24.0 \
                               astroid==1.6.5 \
                               atomicwrites==1.1.5 \
                               aws-xray-sdk==0.95 \
                               backports.functools-lru-cache==1.5 \
                               backports.ssl-match-hostname==3.5.0.1 \
                               backports.tempfile==1.0 \
                               backports.weakref==1.0.post1 \
                               boto==2.49.0 \
                               certifi==2018.8.13 \
                               cffi==1.11.5 \
                               chardet==3.0.4 \
                               configparser==3.5.0 \
                               cookies==2.2.1 \
                               coverage==4.5.1 \
                               cryptography==2.3.1 \
                               docker==3.5.0 \
                               docker-pycreds==0.3.0 \
                               ecdsa==0.13 \
                               enum34==1.1.6 \
                               funcsigs==1.0.2 \
                               future==0.16.0 \
                               idna==2.7 \
                               ipaddress==1.0.22 \
                               isort==4.3.4 \
                               jsondiff==1.1.1 \
                               jsonpickle==0.9.6 \
                               lazy-object-proxy==1.3.1 \
                               mccabe==0.6.1 \
                               mock==2.0.0 \
                               more-itertools==4.3.0 \
                               pathlib2==2.3.2 \
                               pbr==4.2.0 \
                               pluggy==0.7.1 \
                               py==1.5.4 \
                               pyaml==17.12.1 \
                               pycparser==2.18 \
                               pycryptodome==3.6.6 \
                               python-jose==2.0.2 \
                               requests==2.19.1 \
                               responses==0.9.0 \
                               scandir==1.9.0 \
                               singledispatch==3.4.0.3 \
                               urllib3==1.23 \
                               websocket-client==0.51.0 \
                               werkzeug==0.14.1 \
                               wrapt==1.10.11 \
                               xmltodict==0.11.0

# code: processor
COPY base_image_microscopy_processor    /app/base_image_microscopy_processor
COPY czi_processor/czi_processor        /app/czi_processor
COPY czi_processor/run.py               /app/run.py

# code: tests
COPY czi_processor/tests                /app/tests

WORKDIR /app

ENTRYPOINT [""]
