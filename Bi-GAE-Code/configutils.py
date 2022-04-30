import json

def save_config(config,filename):
    config_content = {}
    for key,value in config.items():
        config_content[key] = value
    fw = open(filename,'w',encoding='utf-8')
    dic_json = json.dumps(config_content,ensure_ascii=False,indent=4)
    fw.write(dic_json)
    fw.close()

def load_config(config_file):
    f = open(config_file,encoding='utf-8')
    res = f.read()
    config_content = json.loads(res)
    return  config_content