#!/bin/bash

tmp_dir=/tmp/sentence_transformer

# python download.py


# package the bi-encoder
mar_path=`pwd`
cd $tmp_dir
zip -r $mar_path/pytorch_model.bin 0_Transformer
zip -r $mar_path/pool.zip 1_Pooling
cd -
cp $tmp_dir/config.json $tmp_dir/modules.json .

torch-model-archiver --model-name sentence_xformer --version 1.0 --serialized-file pytorch_model.bin --handler handler.py --extra-files "./config.json,./pool.zip,./modules.json"

# package the cross-encoder
cp -r /tmp/cross_encoder/ ./tmp # need to copy because model archiver cannot access files in root directory

torch-model-archiver --model-name cross_encoder --version 1.0 --serialized-file ./tmp/cross_encoder/pytorch_model.bin --handler ce_handler.py --extra-files "./tmp/cross_encoder/config.json,./tmp/cross_encoder/tokenizer_config.json,./tmp/cross_encoder/merges.txt,./tmp/cross_encoder/special_tokens_map.json,./tmp/cross_encoder/vocab.json"

mkdir model_store
mv sentence_xformer.mar model_store/
mv cross_encoder.mar model_store/