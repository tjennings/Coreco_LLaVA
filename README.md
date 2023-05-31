# Coreco's LLaVA captioning fork

This is a total hack to use LLaVA to caption large image datasets To use it you must: 

* Follow the LLaVA installation instructions to set up this code base, acquire the llama model, and merge their delta. 
* Run the following command: 

```
python3 -m llava.eval.run_llava --path /mnt/van_gogh/model_training/deltron/image_staging/download
```

* resize is supported but optional.  