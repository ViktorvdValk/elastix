FROM ubuntu:20.04

LABEL org.opencontainers.image.documentation="https://github.com/SuperElastix/elastix/wiki"
LABEL org.opencontainers.image.licenses="Apache License Version 2.0"
LABEL modelzoo="https://elastix.lumc.nl/modelzoo/"

# Prepare system packages
RUN apt-get update

# wget is need to get Elastix package
# libgomp1 is required by elastix
RUN apt-get -qq install libgomp1 -y

COPY uploads/bin/* /usr/local/bin/
COPY uploads/lib/* /usr/lib/

COPY uploads/LICENSE /
COPY uploads/NOTICE /

CMD elastix
