#!/bin/bash

tmp_dir=/tmp/sentence_transformer

python download.py

mar_path=`pwd`
cd $tmp_dir
zip -r $mar_path/pytorch_model.bin 0_Transformer
zip -r $mar_path/pool.zip 1_Pooling
cd -
cp $tmp_dir/config.json $tmp_dir/modules.json .

torch-model-archiver --model-name sentence_xformer --version 1.0 --serialized-file pytorch_model.bin --handler handler.py --extra-files "./config.json,./pool.zip,./modules.json"

mkdir model_store
mv sentence_xformer.mar model_store/