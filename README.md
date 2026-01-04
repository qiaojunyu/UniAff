
# UniAff: A Unified Representation of Affordances for Tool Usage and Articulation with Vision-Language Models

## env

``````shell
conda create -n uniaff python=3.9 -y
``````

``````shell
pip install -r requirements.txt
``````
## anno urdf
https://huggingface.co/datasets/qiaojunyu/uniaff

## render finetune data of tools
``````shell
python render_rigid_object.py --debug 0 --ray_tracing 1 --object_type train --file_path xxx  --save_path xxx
``````

``````shell
python process_rigid_object_to_llm_affordance.py  --FPS_sample_number 20000 --down_sample_save_path xxx --render_data_path xxx  --urdf_path_root xxx
``````

## render finetune data of articulated objects

## Fine-tuning implementation
Please refer to  https://github.com/changhaonan/A3VLM/tree/main/model