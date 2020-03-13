gpu=
epochs=10

while [ "$1" != "" ]; do
    case $1 in
        --gpu)                  gpu="--gpu"  ;;
        --epochs )              shift
                                epochs=$1 ;;
        * )                     printf "Invalid arguments."
                                exit 1
    esac
    shift
done

echo "Starting multi param tests"
for model in resnet18 resnet34 resnet50 vgg19; do
  for resize in 0 32 64 128; do
    rm -rf models
    mkdir models
    if [ $resize = 0 ]; then
          python3 main.py --train --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv --model $model --epochs $epochs $gpu --exp-name $model-$resize
    else
      python3 main.py --train --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv --model $model --epochs $epochs --resize $resize $gpu --exp-name $model-$resize
    fi
    mkdir -p models_$resize
    mv models/* models_$resize
    sleep 3
  done
done
