#!/bin/bash

#SBATCH --job-name=m15_xl
#SBATCH --output=./logs/sample-%j.out
#SBATCH --error=./logs/sample-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-user hxu64@jhu.edu
#SBATCH --mail-type end
#SBATCH --partition=brtx6-10
#SBATCH --requeue

LANGS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim,eng'
LANG_PAIRS='nso-eng,run-eng,ssw-eng,ind-eng,msa-eng,isl-eng,nob-eng,fao-eng,slv-eng,tgl-eng,cat-eng,glg-eng,fur-eng,ltz-eng,lim-eng,eng-nso,eng-run,eng-ssw,eng-ind,eng-msa,eng-isl,eng-nob,eng-fao,eng-slv,eng-tgl,eng-cat,eng-glg,eng-fur,eng-ltz,eng-lim'

ADAPTER_FREQ=${1:-1}
BOTTLENECK=${2:--1}
SIZE=${3:-l}
CUSTOMIZE_DESIGN=${4-''}
PRETRAINED_CHECKPOINT=/brtx/606-nvme1/haoranxu/moa_checkpoints/pretrained_m15_size_shard_valid_l/checkpoint_best-shard0.pt

FREQ=4
DATA_DIR=/brtx/606-nvme1/haoranxu/m15_32k
DATA_BIN=${DATA_DIR}/data_bin/shard000/
MAX_UPDATES=25000
ARCH=transformer
MAX_TOKENS=8192

export WANDB_NAME=${SLURM_JOB_ID}_lang_pair_adapter_m15_${ADAPTER_FREQ}_${SIZE}_${CUSTOMIZE_DESIGN}
echo " bottleneck ${BOTTLENECK}; adapter freq ${ADAPTER_FREQ}; model size ${SIZE}; customized design ${CUSTOMIZE_DESIGN}"
sleep 1


if [ ${BOTTLENECK} == "-1" ]; then
    BOTTLENECK=""
else
    BOTTLENECK="--adapter-hidden-dim ${BOTTLENECK}"
fi

if [ ${CUSTOMIZE_DESIGN} ]; then
    CUSTOMIZE_DESIGN="--moa-detail-assign ${CUSTOMIZE_DESIGN}"
fi

if [ ${SIZE} == 'l' ]; then
    LAYER=6
    DIM=512
    FFN_DIM=2048
    HEADS=8
elif [ ${SIZE} == 'xl' ]; then
    LAYER=6
    DIM=1024
    FFN_DIM=4096 
    HEADS=16
elif [ ${SIZE} == 'xxl' ]; then
    LAYER=12
    DIM=1024
    FFN_DIM=4096
    HEADS=16
fi


source ~/.bashrc
conda activate MOA

## Train ## Train (comment for evaluation)
# SRCS='eng'
# TGTS='isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim,eng'

SRCS='eng'
TGTS='tgl,cat,glg,fur,ltz,lim,eng'
# SRCS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim,eng'
# TGTS='eng'

for src in ${SRCS//,/ }; do
    for tgt in ${TGTS//,/ }; do
    if [ ${src} == ${tgt} ]; then
        continue
    fi
    if [ ${src} != 'eng' ] && [ ${tgt} != 'eng' ]; then
        continue
    fi
    SAVE_PATH=/brtx/606-nvme1/haoranxu/moa_checkpoints/lang_pair_adapter_m15_${ADAPTER_FREQ}_${SIZE}_${CUSTOMIZE_DESIGN}/${src}-${tgt}
    mkdir -p ${SAVE_PATH}/results/
    if [ ${src} == 'eng' ]; then
        FSRC=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${src}
        FTGT=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${tgt}
        FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}
        FSTORE=/brtx/606-nvme1/haoranxu/moa_checkpoints/lang_pair_adapter_m15_${ADAPTER_FREQ}_${SIZE}_${CUSTOMIZE_DESIGN}/predict.${src}-${tgt}.${tgt}.bleu
    else 
        FSRC=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${src}
        FTGT=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${tgt}
        FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}
        FSTORE=/brtx/606-nvme1/haoranxu/moa_checkpoints/lang_pair_adapter_m15_${ADAPTER_FREQ}_${SIZE}_${CUSTOMIZE_DESIGN}/predict.${tgt}-${src}.${tgt}.bleu
    fi

    echo ${src}-${tgt}
    
 python train.py  ${DATA_BIN} --arch ${ARCH}  --task translation_multi_simple_epoch \
 --source-lang ${src} --target-lang ${tgt} \
 --moa-type lang_pair --moa-freq ${ADAPTER_FREQ}  ${BOTTLENECK} ${CUSTOMIZE_DESIGN} \
 --lang-pairs ${LANG_PAIRS} --langs ${LANGS} --sampling-method temperature --sampling-temperature 1 --encoder-langtok src --decoder-langtok \
 --encoder-layers ${LAYER} --decoder-layers ${LAYER} --encoder-ffn-embed-dim ${FFN_DIM} --decoder-ffn-embed-dim ${FFN_DIM} \
 --encoder-embed-dim ${DIM} --decoder-embed-dim ${DIM} --encoder-attention-heads ${HEADS} --decoder-attention-heads ${HEADS} --attention-dropout 0.1 --relu-dropout 0.0 \
 --decoder-normalize-before --encoder-normalize-before --share-all-embeddings --max-source-positions 512 --max-target-positions 512 \
 --max-update ${MAX_UPDATES} --update-freq ${FREQ}  --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0004 --stop-min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 --best-checkpoint-metric loss --max-tokens ${MAX_TOKENS}  --validate-interval-updates 100 --save-interval-updates 100 --save-interval 2 --no-epoch-checkpoints \
 --keep-interval-updates 1  --validate-interval 100  --seed 42 --log-format simple --log-interval 100 --fp16 --optimizer adam --min-params-to-wrap 100000000 --max-epoch 50 \
 --save-dir ${SAVE_PATH}  --skip-invalid-size-inputs-valid-test --memory-efficient-fp16  --patience 5 \
 --moa-finetune-from-model ${PRETRAINED_CHECKPOINT}

    fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
        --langs ${LANGS} \
        --lang-pairs ${LANG_PAIRS} \
        --task translation_multi_simple_epoch \
        --is-moa \
        --bpe "sentencepiece" \
        --sacrebleu \
        --encoder-langtok src --decoder-langtok \
        --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.32000.model \
        --moa-finetune-from-model ${PRETRAINED_CHECKPOINT} ${REPLACE_BOS_EOS} \
        --source-lang ${src} --target-lang ${tgt} \
        --batch-size 100 \
        --beam 5 --lenpen 1.0 \
        --no-progress-bar |\
        sort -t '-' -nk 2 | grep -P "^D-" | cut -f 3- > $FOUT
        SACREBLEU_FORMAT=text sacrebleu -tok flores200 $FOUT < ${FTGT} > $FOUT.bleu
    cp $FOUT.bleu ${FSTORE}
    cat ${FSTORE}
    done
done

exit

# # Evaluate
# SRCS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
# tgt=eng
# mkdir -p ${SAVE_PATH}/results
# replication_count=$[ 5 / ${ADAPTER_NUM} ]
# for src in ${SRCS//,/ }; do
#     echo predict $src to $tgt
#     FSRC=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${src}
#     FTGT=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${tgt}
#     FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}

#     fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
#         --langs ${LANGS} \
#         --lang-pairs ${LANG_PAIRS} \
#         --task translation_multi_simple_epoch \
#         --is-moa \
#         --scoring sacrebleu \
#         --sacrebleu \
#         --encoder-langtok src --decoder-langtok \
#         --bpe "sentencepiece" \
#         --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.32000.model \
#         --moa-finetune-from-model ${PRETRAINED_CHECKPOINT} ${REPLACE_BOS_EOS} \
#         --source-lang ${src} --target-lang ${tgt} \
#         --distributed-world-size 5 --distributed-port -1 \
#         --batch-size 100  \
#         --beam 5 --lenpen 1.0 \
#         --model-overrides "{'world_size': 5, 'moa_eval_capacity_token_fraction': 1.0, 'use_moe_pad_mask': False, 'pass_tokens_transformer_layer': False, 'replication_count': ${replication_count}}" \
#         --no-progress-bar |\
#         sort -t '-' -nk 2 | grep -P "^D-" | cut -f 3- > $FOUT
#         SACREBLEU_FORMAT=text sacrebleu -tok flores200 $FOUT < ${FTGT} > $FOUT.bleu
#     cat ${FOUT}.bleu
# done

# TGTS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
# src=eng
# for tgt in ${TGTS//,/ }; do
#     echo predict $src to $tgt
#     FSRC=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${src}
#     FTGT=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${tgt}
#     FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}

#     fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
#         --langs ${LANGS} \
#         --lang-pairs ${LANG_PAIRS} \
#         --task translation_multi_simple_epoch \
#         --is-moa \
#         --bpe "sentencepiece" \
#         --sacrebleu \
#         --encoder-langtok src --decoder-langtok \
#         --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.32000.model \
#         --moa-finetune-from-model ${PRETRAINED_CHECKPOINT} ${REPLACE_BOS_EOS} \
#         --source-lang ${src} --target-lang ${tgt} \
#         --distributed-world-size 5 --distributed-port -1 \
#         --batch-size 100 \
#         --beam 5 --lenpen 1.0 \
#         --model-overrides "{'world_size': 5, 'moa_eval_capacity_token_fraction': 1.0, 'use_moe_pad_mask': False, 'pass_tokens_transformer_layer': False, 'replication_count': ${replication_count}}" \
#         --no-progress-bar |\
#         sort -t '-' -nk 2 | grep -P "^D-" | cut -f 3- > $FOUT
#         SACREBLEU_FORMAT=text sacrebleu -tok flores200 $FOUT < ${FTGT} > $FOUT.bleu
#     cat ${FOUT}.bleu
# done


SAVE_PATH=/brtx/606-nvme1/haoranxu/moa_checkpoints/lang_pair_adapter_m15_${ADAPTER_FREQ}_${SIZE}_${CUSTOMIZE_DESIGN}
# Print
SRCS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
tgt=eng
for src in ${SRCS//,/ }; do
    FOUT=${SAVE_PATH}/predict.${tgt}-${src}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 3
done

# Print
TGTS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
src=eng
for tgt in ${TGTS//,/ }; do
    FOUT=${SAVE_PATH}/predict.${src}-${tgt}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 3
done


echo 'eng->xx'
python ./get_m15_mean.py \
    --input ${SAVE_PATH}/
    
echo 'xx->eng'
python ./get_m15_mean.py \
    --input ${SAVE_PATH} \
    --engtgt 1
