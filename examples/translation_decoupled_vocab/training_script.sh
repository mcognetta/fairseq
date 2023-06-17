#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

# echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
# git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
#BPE_TOKENS=10000
SRC_BPE_TOKENS=3000
TGT_BPE_TOKENS=3000

EXPERIMENT_NAME="test_multiple_loads_3k_3k.de-en"

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi


src=de
tgt=en
lang=de-en
prep=$EXPERIMENT_NAME
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l

    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $tmp/test.$l
done

# TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code
BPE_VOCAB=$prep/vocab
# rm -f $TRAIN
# for l in $src $tgt; do
#     cat $tmp/train.$l >> $TRAIN
# done
# for l in $src $tgt; do
#     echo "learn_BPE for $l"
#     python3 $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.$l > $BPE_CODE.$l
# done

echo "learn_BPE for src: $src"
# python3 $BPEROOT/learn_joint_bpe_and_vocab.py -s $SRC_BPE_TOKENS < $tmp/train.$src > $BPE_CODE.$src
python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $tmp/train.$src -s $SRC_BPE_TOKENS -t -o $BPE_CODE.$src --write-vocabulary $BPE_VOCAB.$src

echo "learn_BPE for tgt: $tgt"
# python3 $BPEROOT/learn_joint_bpe_and_vocab.py -s $TGT_BPE_TOKENS < $tmp/train.$tgt > $BPE_CODE.$tgt
python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $tmp/train.$tgt -s $TGT_BPE_TOKENS -t -o $BPE_CODE.$tgt --write-vocabulary $BPE_VOCAB.$tgt


# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py (${L}) to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE.$L < $tmp/$f > $prep/$f
    done
done

cd ../..

TEXT=examples/translation_decoupled_vocab/$EXPERIMENT_NAME
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --srcdict examples/translation_decoupled_vocab/$EXPERIMENT_NAME/vocab.$src \
    --tgtdict examples/translation_decoupled_vocab/$EXPERIMENT_NAME/vocab.$tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME \
    --workers 20

cp ${TEXT}/train.$src examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME/train.raw.$src
cp ${TEXT}/train.$tgt examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

cp ${TEXT}/vocab.$src ${TEXT}/vocab.$tgt examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME/
cp ${TEXT}/code.$src ${TEXT}/code.$tgt examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME/


sed -i -r 's/(@@ )|(@@ ?$)//g' examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME/train.raw.$src
sed -i -r 's/(@@ )|(@@ ?$)//g' examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

CUDA_VISIBLE_DEVICES=0; nohup fairseq-train  examples/translation_decoupled_vocab/data-bin/$EXPERIMENT_NAME \
                                            --arch transformer_iwslt_de_en \
                                            --share-decoder-input-output-embed \
                                            --optimizer adam --adam-betas '(0.9, 0.98)' \
                                            --clip-norm 0.0 \
                                            --lr 5e-4 \
                                            --lr-scheduler inverse_sqrt \
                                            --warmup-updates 4000 \
                                            --dropout 0.3 \
                                            --weight-decay 0.0001 \
                                            --criterion label_smoothed_cross_entropy \
                                            --label-smoothing 0.1 \
                                            --max-tokens 4096 \
                                            --eval-bleu  \
                                            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
                                            --eval-bleu-detok moses \
                                            --eval-bleu-remove-bpe \
                                            --eval-bleu-print-samples \
                                            --best-checkpoint-metric bleu \
                                            --maximize-best-checkpoint-metric \
                                            --patience 5  \
                                            --save-dir "${EXPERIMENT_NAME}_checkpoints" \
                                            --source-lang=$src \
                                            --target-lang=$tgt \
                                            --task "translation" \
                                            --no-epoch-checkpoints > $EXPERIMENT_NAME.log &