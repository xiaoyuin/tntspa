models=(neural_sparql_machine neural_sparql_machine_bahdanau_attention neural_sparql_machine_luong_attention lstm_luong_wmt_en_de fconv_wmt_en_de transformer_iwslt_de_en wmt16_gnmt_4_layer wmt16_gnmt_8_layer)

d_models=(lstm_bahdanau_attention_bidirectional_encoder transformer_base_single_gpu)

datasets=(monument_600 monument2_1 monument2_2 lc-quad1 dbnqa1)

runs=(run1 run2)

for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for run in ${runs[@]}
        do
            mkdir -p results/$dataset/$run/$model/train
            mkdir -p results/$dataset/$run/$model/eval
        done
    done
done

for dataset in ${datasets[@]}
do
    for model in ${d_models[@]}
    do 
        rm -rf results/$dataset/run1/$model
    done
done