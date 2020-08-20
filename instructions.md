#NVidia-SPADE (pytorch)

## Pre-req

./install.sh (this will install the required packages and download the checkpoint)


## Test
To run an example on a single test image

### coco

python test_image.py --name coco_pretrained --dataset_mode coco --test_file_path FILE --output_dir results

### ade20k

python test_image.py --name ade20k_pretrained --dataset_mode ade20k --test_file_path FILE --output_dir results

To use the CPU add --gpu_ids -1

### OSC

    Receiver 

    python .\osc_receiver.py --name coco_pretrained --dataset_mode coco --output_dir result --osc_send_ip 127.0.0.1 --osc_send_port 5005 --osc_receive_port 5006

    Sender 

    python .\osc_sender.py --file_path FILE --osc_send_ip RECEIVER_IP --osc_send_port 5006 --osc_receive_port 5005