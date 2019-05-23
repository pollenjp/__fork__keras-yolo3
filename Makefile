GPU=0

train : 
	python3 train.py \
		--annotation_filepath="../COCO-Dataset/data/processed/2014/target-train.txt" \
		--train_image_dir_path="../COCO-Dataset/data/raw/2014/train2014" \
		--log_dir="./logs/000/" \
		--anchors_filepath="model_data/yolo_anchors.txt" \
		--weights_filepath='model_data/yolo.h5' \
		--class_filepath="../COCO-Dataset/data/processed/2014/class_names.txt" \
		--input_dim=416 \
		--gpu=${GPU}

predict :
	CUDA_VISIBLE_DEVICES=${GPU} \
		python3 yolo_video.py \
		--anchors_path="model_data/yolo_anchors.txt" \
		--classes_path="../COCO-Dataset/data/processed/2014/class_names.txt" \
		--image \
		--image_dir=./test-images

predict-retrain :
	CUDA_VISIBLE_DEVICES=${GPU} \
		python3 yolo_video.py \
		--model_path=./logs/000/ep072-loss40.694-val_loss42.992.h5 \
		--anchors_path="model_data/yolo_anchors.txt" \
		--classes_path="../COCO-Dataset/data/processed/2014/class_names.txt" \
		--image \
		--image_dir=./test-images


train-bar :
	python3 train.py \
		--annotation_filepath=../foo_bar-detect-fraud/data/processed/obj-det/aiueo/target-bbox-CameraAngle-all.txt \
		--train_image_dir_path=../foo_bar-detect-fraud/data/raw/obj-det/aiueo/ \
		--log_dir="./logs/bar-CameraAngle-all/" \
		--anchors_filepath="model_data/yolo_anchors.txt" \
		--weights_filepath='model_data/yolo.h5' \
		--class_filepath=../foo_bar-detect-fraud/data/processed/obj-det/aiueo/class.names \
		--input_dim=416 \
		--gpu=${GPU}

train-bar_with_no-person :
	python3 train.py \
		--annotation_filepath=../foo_bar-detect-fraud/data/processed/obj-det_with_no-person/target-bbox-CameraAngle-allclass_with_no-person.txt \
		--train_image_dir_path=../foo_bar-detect-fraud/data/processed/obj-det_with_no-person \
		--log_dir="./logs/bar-CameraAngle-all_with_no-person/" \
		--anchors_filepath="model_data/yolo_anchors.txt" \
		--weights_filepath='model_data/yolo.h5' \
		--class_filepath=../foo_bar-detect-fraud/data/processed/obj-det_with_no-person/class_with_no-person.names \
		--input_dim=416 \
		--gpu=${GPU}

predict-bar :
	CUDA_VISIBLE_DEVICES=${GPU} \
		python3 yolo_video.py \
		--model_path=./logs/bar/ep069-loss10.986-val_loss10.867.h5 \
		--anchors_path=model_data/yolo_anchors.txt \
		--classes_path=../foo_bar-detect-fraud/data/processed/obj-det/aiueo/class.names \
		--image \
		--image_dir=../foo_bar-detect-fraud/data/raw/obj-det/aiueo/CameraAngle01/01_46_0_2/Images
