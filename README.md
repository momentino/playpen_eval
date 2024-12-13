# playpen_eval

# Cloning the repository correctly
Since it contains submodules, it's important to initialize them correctly:

```git clone --recurse-submodules https://github.com/momentino/playpen_eval.git```

# Creating the environment  
Create a new conda environment 
```conda create playpen_eval```  
Define a new environment variable with (recommended if you are using models which are gated on HF)  
```conda env config vars set HF_TOKEN=<your token>```  
Run:  
```./install.sh```

# Requirements:
- python==3.10
