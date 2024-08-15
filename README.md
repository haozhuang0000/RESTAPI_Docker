# RESTAPI_Docker

This project involves implementing Hugging Face embedding models with the LangChain ecosystem using a REST API within a Docker environment.

## Why is it useful?

By hosting these embedding models on a GPU server, you can use them from a local PC without a GPU.

## OS

It is highly recommended to run this on a Linux server.

## Docker 

Please download docker first: https://www.docker.com/products/docker-desktop/

`cd RESTAPI_Docker` <br>
`docker build -t embedding . --no-cache` <br>
`docker run --name embedding -p 7777:5000 embedding`

## Usage

After running your embedding container, visit: http://localhost:7777/instruction

for server: simply change localhost to your server's ip address