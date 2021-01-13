
# k2 functions compatible with ESPnet

This project implements some k2 functions compatible with ESPnet. The directory structure is the same with ESPnet.
Currently we support k2 CTC loss.

## k2 CTC Loss

- `K2CTCLoss` is defined in `espnet/nets/pytorch_backend/ctc_graph.py`. 
- The example of usage is in `espnet/nets/pytorch_backend/ctc.py`
- To use `K2CTCLoss` in ESPnet, you should also modify `espnet/bin/asr_train.py` and the training yaml file (`ctc_type: k2`).
