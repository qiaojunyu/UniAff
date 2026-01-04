# env

``````shell
conda create -n uniaff python=3.9 -y
``````

``````shell
pip install -r requirements.txt
``````

# render data of tool
``````shell
python render_rigid_object.py --debug 0 --ray_tracing 1 --object_type train --file_path xxx  --save_path xxx
``````

``````shell
python process_rigid_object_to_llm_affordance.py 
``````