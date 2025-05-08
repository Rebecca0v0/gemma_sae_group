#!/bin/bash

START=0
END=4522815
STEP=200000

for ((i=$START; i<$END; i+=$STEP)); do
  NEXT=$((i+STEP))
  if [ $NEXT -gt $END ]; then
    NEXT=$END
  fi

  echo "Running batch: $i to $NEXT"
  TORCHINDUCTOR_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -m torch.distributed.run --nproc_per_node=3 -- extract_activations.py --start $i --end $NEXT

done