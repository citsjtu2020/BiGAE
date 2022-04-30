from configutils import load_config,save_config


final_datas = load_config("glass_age_male_sim5.json")

t = 0
f_to_index = {}
for k in final_datas['train'].keys():
    f_to_index[k] = t
    t+=1
    # print(k)
    # print(t)
    # t += 1

save_config(f_to_index,"class_to_index.json")

