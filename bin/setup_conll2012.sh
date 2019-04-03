#!/bin/bash

base_data_dir=../bnlp/data

dlx() {
  wget $1/$2
  tar -xvzf $2 -C $base_data_dir
  rm $2
}

conll_url=http://conll.cemantix.org/2012/download
dlx $conll_url conll-2012-train.v4.tar.gz
dlx $conll_url conll-2012-development.v4.tar.gz
dlx $conll_url/test conll-2012-test-key.tar.gz
dlx $conll_url/test conll-2012-test-official.v9.tar.gz

dlx $conll_url conll-2012-scripts.v3.tar.gz

dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mv $base_data_dir/reference-coreference-scorers $base_data_dir/conll-2012/scorer

# You need change ontonotes_path to your
ontonotes_path=/data/ontonotes-release-5.0
bash $base_data_dir/conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data $base_data_dir/conll-2012

function compile_partition() {
    rm -f $2.$5.$3$4
    cat $base_data_dir/conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >> $2.$5.$3$4
}

function compile_language() {
    compile_partition development dev v4 _gold_conll $1
    compile_partition train train v4 _gold_conll $1
    compile_partition test test v4 _gold_conll $1
}

function mv_data() {
    mv dev.$1.v4_gold_conll $base_data_dir/
    mv train.$1.v4_gold_conll $base_data_dir/
    mv test.$1.v4_gold_conll $base_data_dir/
}

compile_language english
compile_language chinese
compile_language arabic

mv_data english
mv_data chinese
mv_data arabic