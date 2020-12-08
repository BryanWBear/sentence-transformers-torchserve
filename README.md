# sentence-transformers-torchserve

1. docker build -t <IMAGE_NAME> .
2. docker run --rm -it \                                          
-p 3000:8080 -p 3001:8081 <IMAGE_NAME> \
torchserve --start --model-store model_store --models all