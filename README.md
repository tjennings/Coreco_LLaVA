# Coreco's LLaVA captioning fork

This is a total hack to use LLaVA to caption large image datasets To use it you must: 

* Follow the LLaVA installation instructions to set up this code base, acquire the llama model, and merge their delta. 
* Run the following command: 

```
python3 -m llava.eval.run_llava --path /path/to/images --model_path /path/to/model
```

Full options: 

```
usage: described [-h] --path PATH [--overwrite] [--model_name MODEL_NAME] [--prefix PREFIX] [--suffix SUFFIX] [--resize RESIZE]

options:
  -h, --help            show this help message and exit
  --path PATH           Path to images to be captioned
  --overwrite           Overwrite existing captions
  --model_name MODEL_NAME
                        a string applied at the beginning of each caption
  --prefix PREFIX       a string applied at the beginning of each caption
  --suffix SUFFIX       a string applied at the end of each caption
  --resize RESIZE       additionally, resize and save the image where the longest side is the provided maximum

```
